# -*- coding: utf-8 -*-
import sys

import tensorflow_recommenders as tfrs

import tensorflow as tf
import numpy as np

import pandas as pd
import pymysql


def recommend_recipes(user_id):

    #데이터베이스 연결
    connection = pymysql.connect(
        user="kawa",
        password="0000",
        host="localhost",
        port=3307,
        database="board"
    )

    cursor = connection.cursor()

    # 이 부분에서 SQL 쿼리를 실행
    #게시글 데이터베이스 연결 및 불러오기
    cursor.execute("""
                SELECT * FROM tbl_board WHERE status = 'Y';
             """)

    result1 = cursor.fetchall()

    # 나머지 코드 실행
    recipes = pd.DataFrame(result1)
    #columns로 몇번째 열에 어떤 컬럼이 있는지 명시해주기
    recipes.rename(columns={0: 'bno'}, inplace=True)
    recipes.rename(columns={1: 'title'}, inplace=True)
    recipes.rename(columns={2: 'content'}, inplace=True)
    recipes.rename(columns={5: 'viewCnt'}, inplace=True)

    # 텍스트 데이터 정제
    recipes['content'] = recipes['content'].str.replace(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]', '', regex=True)

    #후기 데이터베이스
    cursor.execute("""
                        SELECT * FROM tbl_post;
                        """)

    result1 = cursor.fetchall()
    post = pd.DataFrame(result1)
    post.rename(columns = {1 : 'bno'}, inplace = True)
    post.rename(columns = {7 : 'userId'}, inplace = True)

    #좋아요(찜) 데이터베이스
    cursor.execute("""
                    SELECT * FROM tbl_like;
                    """)

    result1 = cursor.fetchall()
    like = pd.DataFrame(result1)
    like.rename(columns = {1 : 'bno'}, inplace = True)
    like.rename(columns = {2 : 'userId'}, inplace = True)


    # 전체 게시글 데이터 개수 가져오기
    cursor.execute("""
        SELECT COUNT(*) FROM tbl_board WHERE status = 'Y';
    """)
    result = cursor.fetchone()
    total_data_count = result[0]


    # 'tbl_post.csv'에서 'bno'와 'userId' 열을 선택
    # 후기 데이터프레임 형태 변환
    post['userId'] = post['userId'].astype(str)
    post = post[['bno', 'userId']]

    # 'tbl_like.csv'에서 'bno'와 'userId' 열을 선택
    # 찜 데이터프레임 형태 변환
    like['userId'] = like['userId'].astype(str)
    like = like[['bno', 'userId']]

    # 'tbl_board.csv'에서 'bno'와 'title' 열을 선택
    # 게시글 데이터프레임 형태 변환
    recipes['title'] = recipes['title'].astype(str)
    recipes['content'] = recipes['content'].astype(str)
    recipes = recipes[['bno', 'title', 'content', 'viewCnt']]


    # 'bno'를 기준으로 두 데이터 프레임을 병합
    # 후기와 게시글 데이터 병합
    merged_data1 = post.merge(recipes, on='bno', how='inner')
    # 찜과 게시글 데이터 병합
    merged_data2 = like.merge(recipes, on='bno', how='inner')

    # 두 번째 병합: bno와 userId 중 하나라도 다른 경우 추가
    merged_data = pd.concat([merged_data1, merged_data2])

    #drop_duplicates 함수를 사용하여 중복된 행을 제거
    merged_data = merged_data.drop_duplicates(subset=['bno', 'userId'])

    # # 필요한 열 선택
    post = merged_data[['bno', 'content', 'userId']]
    recipes = recipes[['bno', 'title', 'content', 'viewCnt']]

    def pandas_to_dataset(dataframe, target_column_name):
        dataframe = dataframe.copy()
        labels = dataframe.pop(target_column_name)  # 타겟 열 선택
        dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        return dataset

    # 후기 데이터셋 생성
    ratings_dataset = pandas_to_dataset(post,'userId')
    # 레시피 데이터셋 생성
    recipes_dataset = pandas_to_dataset(recipes, 'content')

    # Create a function to convert a row to the desired format
    def map_features(x,y):
        return {
            "content": x["content"],
            "userId": y
        }

    # 데이터셋의 각 요소에 map_features 함수를 적용하여 새로운 형태의 데이터셋 생성
    ratings_dataset = ratings_dataset.map(map_features)
    # recipes 데이터프레임의 'content' 열을 tf.data.Dataset으로 변환
    recipes_dataset = tf.data.Dataset.from_tensor_slices(recipes['content'])

    # 사용자 ID와 게시물 content에 대한 Vocabulary 생성
    user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    user_ids_vocabulary.adapt(ratings_dataset.map(lambda x: x['userId']))

    recipes_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
    recipes_titles_vocabulary.adapt(recipes_dataset)



    class RecipesLensModel(tfrs.Model):
        def __init__(self, user_model, recipes_model, task):
            super(RecipesLensModel, self).__init__()

            # Set up user and recipes representations.
            self.user_model = user_model
            self.recipes_model = recipes_model

            # Set up a retrieval task.
            self.task = task

        def compute_loss(self, features, training=False):

            # Define how the loss is computed.
            user_embeddings = self.user_model(features["userId"])
            recipes_embeddings = self.recipes_model(features["content"])

            return self.task(user_embeddings, recipes_embeddings)

    # 사용자 모델 정의
    user_model = tf.keras.Sequential([
        user_ids_vocabulary,
        tf.keras.layers.Embedding(input_dim=user_ids_vocabulary.vocab_size(), output_dim=64)
    ])

    # 레시피 모델 정의
    recipes_model = tf.keras.Sequential([
        recipes_titles_vocabulary,
        tf.keras.layers.Embedding(input_dim=recipes_titles_vocabulary.vocab_size(), output_dim=64)

    ])

    # 추천 태스크 정의
    task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
        recipes_dataset.batch(128).map(recipes_model)
    ))

    # 추천 모델 정의
    model = RecipesLensModel(user_model, recipes_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

    # 모델 훈련
    model.fit(ratings_dataset.batch(total_data_count), epochs=5, verbose=0)
    userId = user_id  # 사용자 ID 받아오기


    # 기존에 작성한 게시글의 제목을 가져옴
    user_titles = merged_data[merged_data['userId'] == userId]['content'].unique()

    if len(user_titles) == 0:
        # 추천할 게시물이 없는 경우, 인기 있는 게시물을 추천(조회수 높은 게시물)
        num_recommendations = 5
        popular_recipes = recipes.sort_values(by='viewCnt', ascending=False)
        top_recommendations = popular_recipes.head(num_recommendations)
        bno_list = top_recommendations['bno'].tolist()

    else:
        recipes_dataset = recipes_dataset.filter(
            lambda title: tf.math.logical_not(tf.math.reduce_any(tf.math.equal(title, user_titles)))
        )

    # BruteForce 인덱스를 생성하고 데이터셋으로부터 인덱싱함
        index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
        index.index_from_dataset(
            recipes_dataset.batch(100).map(lambda title: (title, model.recipes_model(title)))
        )

        # 사용자에게 추천합니다. 중복을 피하기 위해 루프를 사용함
        recommended_titles = set()
        user_id = np.array([userId], dtype=object)
        i = 0
        while len(recommended_titles) < 5:  # 5개의 서로 다른 게시글 추천
            _, titles = index(user_id)
            for title in titles[0, i:i+1]:  # 다음 추천
                title_text = title.numpy().decode('utf-8')
                if title_text not in recommended_titles:  # 추천된 게시물중에 중복된게 없는지 확인
                    recommended_titles.add(title_text)
                if len(recommended_titles) == 5:  # 원하는 수의 추천을 얻었다면 루프 종료
                    break
            i += 1  # 다음 추천을 위해 인덱스를 증가시킴

        # 레시피 'content'에 대응하는 'bno'를 찾아서 리스트에 저장
        bno_dict = dict(zip(recipes['content'], recipes['bno']))
        bno_list = [bno_dict.get(title, None) for title in recommended_titles]

    # 'bno' 리스트 출력
    for bno in bno_list:
        if bno is not None:
            print(bno)

if __name__ == "__main__":
    # 명령줄 인수에서 userId 값을 읽어옴
    if len(sys.argv) < 2:
        print("Usage: python recommend.py <userId>")
        sys.exit(1)
    user_id = sys.argv[1]
    recommend_recipes(user_id)