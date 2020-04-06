## Project: Caesar Cipher Engine

**Project description:** Builds a simple machine that takes in a string, and return that string as an encrypted message built in the same manner as the Caesar Cipher:
<img src="images/cesar.png?raw=true"/>
Where every encrypted string can only be decrypted if we know the 'offset', i.e. how far would we step the letters in the encrypted message to decrypt it.

This project is built on Python 3.

### 1. Build the Decoder Engine

First, we build the decoder engine as a function to take in the encrypted message, the offset, and the list of alphabet string (in this case, lowercase a-z's). After that, we prepare an empty string to be returned when the decoder engine finishes. Then, we convert the encrypted message to lowercases in order for easier parsing:

```python
message1 = "y wej oekh cuiiqwu!"
alphabet_string = "abcdefghijklmnopqrstuvwxyz"
def decoder_engine(message, alphabet, offset):
    new_string = ""
    converted_message = message.lower()
```

Then, we iterate through each letter of the converted message. If the letter iterated is in our alphabet list, we get the index of that letter, and substitute it with the appropriate index found by adding it with the offset. In other words, we 'shift' the letters by the offset given. Lastly, we add that letter to our empty string.

```python
    for x in converted_message:
        if x in alphabet:
            locate_index = alphabet.find(x)
            new_index = (locate_index + (offset)) % 26
            new_character = alphabet_string[new_index]
            new_string += new_character
```

If the letter iterated is not in alphabet (i.e. special symbols, spaces, etc.), they will not undergo substitution. Rather, they are simply added to the empty string we created earlier.
```python
        elif x not in alphabet:
            new_string += x
    return(new_string)
```
Full code:
```python
message1 = "y wej oekh cuiiqwu!"
alphabet_string = "abcdefghijklmnopqrstuvwxyz"
def decoder_engine(message, alphabet, offset):
    new_string = ""
    converted_message = message.lower()
    for x in converted_message:
        if x in alphabet:
            locate_index = alphabet.find(x)
            new_index = (locate_index + (offset)) % 26
            new_character = alphabet_string[new_index]
            new_string += new_character
        elif x not in alphabet:
            new_string += x
    return(new_string)
```
If we call:
```python3
decoder_engine(message1, alphabet_string, 10)
```
Which means we want to decrypt the string `"y wej oekh cuiiqwu!"` with an offset of 10, the engine would return:
`'i got your message!'`
### 2. Assess assumptions on which statistical inference will be based

```javascript
if (isAwesome){
  return true
}
```

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
