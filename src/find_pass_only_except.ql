import python

from ExceptStmt ex
where forall(Stmt s | s = ex.getAStmt() | s instanceof Pass)
select ex, 
       ex.getLocation().getFile().getRelativePath() + ":" + 
       ex.getLocation().getStartLine().toString(),
       "Empty error handler (pass-only except block)"
