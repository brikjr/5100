import ollama
response = ollama.chat(model='llama3.2-vision', messages=[
  {
    'role': 'user',
    'content': 'Create a JSON with the title and author of each book in the format "title": [title], "author": [author]',
    'images': ['SampleImages/12BooksMultipleDirectionsCrop.jpg']
  },
])
response = response['message']['content']
print(response)
listOfBooks = response.split("[")[1]
listOfBooks = response.split("]")[0]
print(listOfBooks)

