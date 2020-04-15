## Project: Vigenère Cipher Engine

**Project description:** Builds a complicated encryption/decryption engine based on the Vigenère cipher;  a polyalphabetic substitution cipher, as opposed to the Caesar Cipher which was a monoalphabetic substitution cipher. What this means is that opposed to having a single shift that is applied to every letter, the Vigenère Cipher has a different shift for each individual letter.

The 'shifts' are represented through given keywords, which makes this type of cryptography harder to break. 
I built the engine based on the video below:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=SkJcmCaHqS0
" target="_blank"><img src="http://img.youtube.com/vi/SkJcmCaHqS0/0.jpg" 
alt="Vigenère Cipher" width="240" height="180" border="10" /></a>

This project is built on Python 3.

### 1. Getting started

We would like to know which alphabet belongs to which index. This, whilst not directly applied to the function we're gonna build, can provide a better understanding of the groundwork used in building this engine.
```python
alphabet_string = "abcdefghijklmnopqrstuvwxyz"
index_list = [alphabet_string.find(x) for x in alphabet_string]
print(list(zip(index_list, alphabet_string)))
```
This will print a list of alphabets assigned with their respective indices:
```
[(0, 'a'), (1, 'b'), (2, 'c'), (3, 'd'), (4, 'e'), (5, 'f'), (6, 'g'), (7, 'h'), (8, 'i'), (9, 'j'), (10, 'k'), (11, 'l'), (12, 'm'), (13, 'n'), (14, 'o'), (15, 'p'), (16, 'q'), (17, 'r'), (18, 's'), (19, 't'), (20, 'u'), (21, 'v'), (22, 'w'), (23, 'x'), (24, 'y'), (25, 'z')]
```

### 2. Build the Encoder Engine

At first, we need to declare the keyword that we're gonna use throughout the build. We also need to import ```cycle``` from ```itertools```:

```python
from itertools import cycle
keyword1 = "dog"
```

Then, after we've defined the function, we mention all the possible punctuations that appears in the message we're gonna encrypt--so that the function doesn't substitute non-alphabetical characters by mistake. Then, we're gonna put on a bunch of empty lists and empty strings that we're gonna use temporarily while solving the encryption:

```python
def encoder_engine_2(message, alphabet, keyword):
    punctuation = "!?., "
    old_ind = []
    key_ind = []
    new_ind = []
    newStr = ""
    codeStr = ""
    keywordCycle = cycle(keyword)
```

The ```cycle``` object can only be used with the ```next(object)``` command, but we'll get to it later. Now, we need to iterate all the letters from the message and check whether it's alphabetical. If it isn't, we don't wanna bother ourselves so we can just add that to the empty string we created earlier. Otherwise, we want to convert it to the keyword representation of the message:
```python
    for x in message4:
        if x in alphabet:
            newStr += next(keywordCycle)
            old_ind.append(alphabet.find(x))
        elif x in punctuation:
            newStr += x
            old_ind.append(punctuation.find(x) + 90)
```

```old_ind``` is now:
```
[1, 0, 17, 17, 24, 94, 8, 18, 94, 19, 7, 4, 
94, 18, 15, 24, 93, 94, 22, 0, 19, 2, 7, 94, 
14, 20, 19, 94, 5, 14, 17, 94, 7, 8, 12, 90]
```

```newStr```, the keyword representation of the message, is now:
```
dogdo gd ogd ogd, ogdog dog dog dog!
```

Then, we iterate through ```new_str``` to convert that again into the fully-encrypted message:
```python
    for y in newStr:
        if y in alphabet:
            key_ind.append(alphabet.find(y))
        elif y not in alphabet:
            key_ind.append(punctuation.find(y) + 90)
```

If the letter we iterated isn't an alphabet, I added the index by 90 for easier debugging--we can see later that the non-alphabetical characters have a very large index number. 

```key_ind```, the index representation of ```new_Str```, now appears like this:
```
[3, 14, 6, 3, 14, 94, 6, 3, 94, 14, 6, 3, 
94, 14, 6, 3, 93, 94, 14, 6, 3, 14, 6, 94, 
3, 14, 6, 94, 3, 14, 6, 94, 3, 14, 6, 90]
```

The outlier indices at 90-94 represents a non-alphabetical character.
For the final step, I added each element of ```key_ind``` with the corresponding element from ```old_ind```. This is the whole idea of 'shift' that we're talking about, as each letter would have different offsets based on the keyword given. Lastly, we convert the results in ```new_ind``` to the fully-encrypted message:

```python
    for i in range(len(key_ind)):
        new_ind.append(key_ind[i] + old_ind[i])
    for z in new_ind:
        if z < 80:
            codeStr += alphabet[z % 26]
        else:
            codeStr += punctuation[int(z/2) - 90]
```

What I did there was check whether the indices are outliers or not; if they are, then I subtract them by 90 to get the initial value, before converting it to the corresponding punctuation. Otherwise, I simply convert them to the alphabet.

When I call ```encoder_engine_2("barry is the spy, watch out for him!", alphabet_string2, keyword1)```, this will return:
```
eoxum ov hnh gvb, kgwqn riz icx kws!
```
Full code:
```python
message4 = "barry is the spy, watch out for him!"
keyword1 = "dog"
alphabet_string2 = "abcdefghijklmnopqrstuvwxyz"
def encoder_engine_2(message, alphabet, keyword):
    punctuation = "!?., "
    old_ind = []
    key_ind = []
    new_ind = []
    newStr = ""
    codeStr = ""
    keywordCycle = cycle(keyword)
    for x in message4:
        if x in alphabet:
            newStr += next(keywordCycle)
            old_ind.append(alphabet.find(x))
        elif x in punctuation:
            newStr += x
            old_ind.append(punctuation.find(x) + 90)
    for y in newStr:
        if y in alphabet:
            key_ind.append(alphabet.find(y))
        elif y not in alphabet:
            key_ind.append(punctuation.find(y) + 90)
    for i in range(len(key_ind)):
        new_ind.append(key_ind[i] + old_ind[i])
    for z in new_ind:
        if z < 80:
            codeStr += alphabet[z % 26]
        else:
            codeStr += punctuation[int(z/2) - 90]
```
### 3. Build the Decoder Engine

Basically we reverse the logic we used during Step 2. In other words, we unpack the coded message into separate indices, then we convert those indices toward the ones belonging to the decrypted message. 

Full code:
```python
message4 = "dfc aruw fsti gr vjtwhr wznj? vmph otis! cbx swv jipreneo uhllj kpi rahjib eg fjdkwkedhmp!"
keyword1 = "friends"
def decoder_engine_3(message, alphabet, keyword):
    punctuation = "!?,. "
    old_ind = []
    key_ind = []
    new_ind = []
    newStr = ""
    codeStr = ""
    keywordCycle = cycle(keyword)
    for x in message4:
        if x in alphabet:
            newStr += next(keywordCycle)
            old_ind.append(alphabet.find(x))
        elif x in punctuation:
            newStr += x
            old_ind.append(punctuation.find(x) + 90)
    for y in newStr:
        if y in alphabet:
            key_ind.append(alphabet.find(y))
        elif y not in alphabet:
            key_ind.append(punctuation.find(y) + 90)
    for (a, b) in zip(old_ind, key_ind):
        if a < 80 and b < 80:
            new_ind.append(a-b)
        else:
            new_ind.append(a)
    for z in new_ind:
        if z < 80:
            codeStr += alphabet[z % 26]
        else:
            codeStr += punctuation[z-90]
decoder_engine_3(message4, alphabet_string, keyword1)
```
**the encrypted message**:
dfc aruw fsti gr vjtwhr wznj? vmph otis! cbx swv jipreneo uhllj kpi rahjib eg fjdkwkedhmp!

**the old index**:
\[3, 5, 2, 94, 0, 17, 20, 22, 94, 5, 18, 19, 8, 94, 6, 17, 94, 21, 9, 19, 22, 7, 17, 94, 22, 25, 13, 9, 91, 94, 21, 12, 15, 7, 94, 14, 19, 8, 18, 90, 94, 2, 1, 23, 94, 18, 22, 21, 94, 9, 8, 15, 17, 4, 13, 4, 14, 94, 20, 7, 11, 11, 9, 94, 10, 15, 8, 94, 17, 0, 7, 9, 8, 1, 94, 4, 6, 94, 5, 9, 3, 10, 22, 10, 4, 3, 7, 12, 15, 90]

**the keyword representation**:
fri ends frie nd sfrien dsfr? iend sfri! end sfr iendsfri endsf rie ndsfri en dsfriendsfr!

**the key index**:
\[5, 17, 8, 94, 4, 13, 3, 18, 94, 5, 17, 8, 4, 94, 13, 3, 94, 18, 5, 17, 8, 4, 13, 94, 3, 18, 5, 17, 91, 94, 8, 4, 13, 3, 94, 18, 5, 17, 8, 90, 94, 4, 13, 3, 94, 18, 5, 17, 94, 8, 4, 13, 3, 18, 5, 17, 8, 94, 4, 13, 3, 18, 5, 94, 17, 8, 4, 94, 13, 3, 18, 5, 17, 8, 94, 4, 13, 94, 3, 18, 5, 17, 8, 4, 13, 3, 18, 5, 17, 90]

**the new index**:
\[-2, -12, -6, 94, -4, 4, 17, 4, 94, 0, 1, 11, 4, 94, -7, 14, 94, 3, 4, 2, 14, 3, 4, 94, 19, 7, 8, -8, 91, 94, 13, 8, 2, 4, 94, -4, 14, -9, 10, 90, 94, -2, -12, 20, 94, 0, 17, 4, 94, 1, 4, 2, 14, -14, 8, -13, 6, 94, 16, -6, 8, -7, 4, 94, -7, 7, 4, 94, 4, -3, -11, 4, -9, -7, 94, 0, -7, 94, 2, -9, -2, -7, 14, 6, -9, 0, -11, 7, -2, 90]

**the decoded message**:
you were able to decode this? nice work! you are becoming quite the expert at crytography!

