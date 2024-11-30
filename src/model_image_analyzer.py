import ollama
def generate_resopnse(image):
  response = ollama.chat(model='llama3.2-vision', messages=[
    {
      'role': 'user',
      'content': 'List the titles of all the books in the image. List the book titles in quotes. Do not include any other text in the response, only the book titles in quotes.',
      'images': [image]
    },
  ])
  response = response['message']['content']
  return response

'''
Function: titles_to_list
  This function takes in the string response from the Llama model and converts the response to a list of titles. This
  function helps to handle some of the inconsistency in the LLM response. The output is list with all quotations removed
  and the title of the book in each entry. 
Parameters:
  response: The string response from the LLM that has the titles of the books in quotation marks. 
Output:
  A list where each entry is the title of the book.
'''
def titles_to_list(response):
    # Setup a flag to account for if a quotation has been seen
    quote_seen = 0
    # Setup an empty string for the title and an empty list for the return value
    current_title = ""
    title_list = []
    # For each character in the response string
    for char in response:
        # If no starting quote has been seen yet and the current char is a quote
        if quote_seen == 0 and char == '"':
            # Set the quote_seen to true to start recording
            quote_seen = 1
        # If the starting quote has been seen and the current char is a quote
        elif quote_seen == 1 and char == '"':
            # Set the quote seen to false, add the title to the list and reset the string
            quote_seen = 0
            title_list.append(current_title)
            current_title = ""
        # If the starting quote is seen then append the current char to the title string
        elif quote_seen == 1:
            current_title += char
    # Remove duplicates using a set
    unique_list = list(set(title_list))
    return unique_list

def processImageGetList(image):
    return titles_to_list(generate_resopnse(image))