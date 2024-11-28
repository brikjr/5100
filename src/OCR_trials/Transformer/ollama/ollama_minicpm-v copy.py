import ollama
response = ollama.chat(model='minicpm-v', messages=[
  {
    'role': 'user',
    'content': 'List the titles and authors of the books in this image in the following format: [title] - [author] and only list the book titles your sure about.',
    'images': ['../5BooksSameDirections.jpg']
  },
])
print(response['message']['content'])