import python
from ExceptStmt ex


where 
  (not exists(ex.getType()) or 
   ex.getType().(Name).getId() = "BaseException" or 
   ex.getType().(Name).getId() = "Exception") and
  exists(Stmt s | s = ex.getAStmt() |
    s.(ExprStmt).getValue().(Call).getFunc().(Attribute).getName() = "exit" or
    s.(ExprStmt).getValue().(Call).getFunc().(Name).getId() = "exit" or
    s.(ExprStmt).getValue().(Call).getFunc().(Name).getId() = "quit" or
    (s instanceof Raise and not exists(s.(Raise).getException()))
  )
select ex,
       ex.getLocation().getFile().getRelativePath() + ":" + 
       ex.getLocation().getStartLine().toString(),
       "Error handler over-catches exceptions and aborts"