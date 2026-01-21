import random
from collections import Counter

random.seed(42)

N = 500
with open("python_projects.txt") as f:
    projects = f.read().splitlines()

subset = random.sample(projects, N)

with open("python_projects_subset.txt", "w") as f:
    f.write("\n".join(subset))

llms = [p.split("/")[2] for p in subset]  
print(Counter(llms))
