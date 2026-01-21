import python

from ExceptStmt ex, Comment c
where
  c.getLocation().getStartLine() >= ex.getLocation().getStartLine() and
  c.getLocation().getEndLine()   <= ex.getLocation().getEndLine() and
(
    c.getText().matches("(?i).*TODO.*") or
    c.getText().matches("(?i).*FIXME.*")
  )

select ex,
  ex.getLocation().getFile().getRelativePath() + ":" +
  ex.getLocation().getStartLine().toString(),
  "Exception handler contains TODO/FIXME comment"
