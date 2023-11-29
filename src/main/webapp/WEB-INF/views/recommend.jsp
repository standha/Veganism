<%--
  Created by IntelliJ IDEA.
  User: myyun
  Date: 2023-10-30
  Time: 오후 3:50
  To change this template use File | Settings | File Templates.
--%>
<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
<head>
    <title>Title</title>
</head>
<body>

</body>
</html>
import org.python.util.PythonInterpreter;

public class main {

private static PythonInterpreter interpreter;

public static void main(String[] args) {

interpreter = new PythonInterpreter();
interpreter.execfile("test.py");
interpreter.exec("print(sum(7,8))");

}

}
