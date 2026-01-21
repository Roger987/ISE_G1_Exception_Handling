import python
from ExceptStmt ex
where 
  forall(Stmt s | s = ex.getAStmt() | 
    s.(ExprStmt).getValue().(Call).getFunc().(Attribute).getName() in ["info", "warning", "error", "debug", "exception"] or
    s.(ExprStmt).getValue().(Call).getFunc().(Name).getId() = "print"
  ) and
  exists(ex.getAStmt())
select ex,
       ex.getLocation().getFile().getRelativePath() + ":" + 
       ex.getLocation().getStartLine().toString(),
       "Exception handler only logs without addressing the issue"