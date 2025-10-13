#File Handling
with open("data.txt", "w") as f:
  f.write("Hello, World!")
with open("data.txt", "r") as f:
  print(f.read())