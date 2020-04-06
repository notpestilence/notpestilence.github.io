## Project: Caesar Cipher Engine

**Project description:** Builds a simple machine that takes in a string, and return that string as an encrypted message built in the same manner as the Caesar Cipher:

<img src="images/cesar.png"/>

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
```python
decoder_engine(message1, alphabet_string, 10)
```
Which means we want to decrypt the string `"y wej oekh cuiiqwu!"` with an offset of 10, the engine would return:
`'i got your message!'`
### 2. Build the Encoder Engine

To build the encoder engine, we simply reverse the logic from Step 1. The only difference was instead of adding the offset, we actually want to subtract it. This is done to 'take backwards step' instead of moving forward with the offset like we did in the previous code.

```python
message2 = "I got your message!"
def encoder_engine(message, alphabet, offset):
    new_string = ""
    converted_message = message.lower()
    for x in converted_message:
        if x in alphabet:
            locate_index = alphabet.find(x)
            new_index = (locate_index - (offset)) % 26
            new_character = alphabet_string[new_index]
            new_string += new_character
        elif x not in alphabet:
            new_string += x
    return (new_string)
```
Thus, if we call:
```encoder_engine(message2, alphabet_string, 10)```
This will return:
```"y wej oekh cuiiqwu!"```
We can see this was the original message we take as an input during Step 1 with the same offset value.

### 3. Decoding Caesar Cipher without the Offset Value

We can do this by iterating the number of function calls within the alphabet range, which is 26. Rationally speaking, the offset value can't be larger than 25. If the offset is, say, 26, the offset value is essentially zero because it will shift into the same letter as before.

If the offset is 27, then essentially it will only shift once. This explains the modulo operator in every function in steps 1 and 2.
```python
message3 = "vhfinmxkl atox kxgwxkxw tee hy maxlx hew vbiaxkl tl hulhexmx. px'ee atox mh kxteer lmxi ni hnk ztfx by px ptgm mh dxxi hnk fxlltzxl ltyx."
def decoder_engine_2(message, alphabet):
    new_string = ""
    converted_message = message.lower()
    for y in range(26):
        for x in converted_message:
            if x in alphabet:
                locate_index = alphabet.find(x)
                new_index = (locate_index + (y)) % 26
                new_character = alphabet_string[new_index]
                new_string += new_character
            elif x not in alphabet:
                new_string += x
```
This is yet to be finished, if we return ```new_string``` in this condition, the results are somewhat clunky and arduous to comprehend:
```
vhfinmxkl atox kxgwxkxw tee hy maxlx hew vbiaxkl tl hulhexmx. px'ee atox mh kxteer lmxi ni hnk ztfx by px ptgm mh dxxi hnk fxlltzxl ltyx.

vhfinmxkl atox kxgwxkxw tee hy maxlx hew vbiaxkl tl hulhexmx. px'ee atox mh kxteer lmxi ni hnk ztfx by px ptgm mh dxxi hnk fxlltzxl ltyx.wigjonylm bupy lyhxylyx uff iz nbymy ifx wcjbylm um ivmifyny. qy'ff bupy ni lyuffs mnyj oj iol augy cz qy quhn ni eyyj iol gymmuaym muzy.

vhfinmxkl atox kxgwxkxw tee hy maxlx hew vbiaxkl tl hulhexmx. px'ee atox mh kxteer lmxi ni hnk ztfx by px ptgm mh dxxi hnk fxlltzxl ltyx.wigjonylm bupy lyhxylyx uff iz nbymy ifx wcjbylm um ivmifyny. qy'ff bupy ni lyuffs mnyj oj iol augy cz qy quhn ni eyyj iol gymmuaym muzy.xjhkpozmn cvqz mziyzmzy vgg ja ocznz jgy xdkczmn vn jwnjgzoz. rz'gg cvqz oj mzvggt nozk pk jpm bvhz da rz rvio oj fzzk jpm hznnvbzn nvaz.

vhfinmxkl atox kxgwxkxw tee hy maxlx hew vbiaxkl tl hulhexmx. px'ee atox mh kxteer lmxi ni hnk ztfx by px ptgm mh dxxi hnk fxlltzxl ltyx.wigjonylm bupy lyhxylyx uff iz nbymy ifx wcjbylm um ivmifyny. qy'ff bupy ni lyuffs mnyj oj iol augy cz qy quhn ni eyyj iol gymmuaym muzy.xjhkpozmn cvqz mziyzmzy vgg ja ocznz jgy xdkczmn vn jwnjgzoz. rz'gg cvqz oj mzvggt nozk pk jpm bvhz da rz rvio oj fzzk jpm hznnvbzn nvaz.ykilqpano dwra najzanaz whh kb pdaoa khz yeldano wo kxokhapa. sa'hh dwra pk nawhhu opal ql kqn cwia eb sa swjp pk gaal kqn iaoowcao owba.

vhfinmxkl atox kxgwxkxw tee hy maxlx hew vbiaxkl tl hulhexmx. px'ee atox mh kxteer lmxi ni hnk ztfx by px ptgm mh dxxi hnk fxlltzxl ltyx.wigjonylm bupy lyhxylyx uff iz nbymy ifx wcjbylm um ivmifyny. qy'ff bupy ni lyuffs mnyj oj iol augy cz qy quhn ni eyyj iol gymmuaym muzy.xjhkpozmn cvqz mziyzmzy vgg ja ocznz jgy xdkczmn vn jwnjgzoz. rz'gg cvqz oj mzvggt nozk pk jpm bvhz da rz rvio oj fzzk jpm hznnvbzn nvaz.ykilqpano dwra najzanaz whh kb pdaoa khz yeldano wo kxokhapa. sa'hh dwra pk nawhhu opal ql kqn cwia eb sa swjp pk gaal kqn iaoowcao owba.zljmrqbop exsb obkaboba xii lc qebpb lia zfmebop xp lyplibqb. tb'ii exsb ql obxiiv pqbm rm lro dxjb fc tb txkq ql hbbm lro jbppxdbp pxcb...
```
...and so on. Thus, we need to trim every finished result by its corresponding iteration number. In other words, if the function undergoes the first iteration, it should only return the results of the first offset. If the function undergoes the 5th iteration, it should only return the results of the fifth offset.

We can do this by:
```
        if len(new_string) <= len(message):
            print(new_string + "\n")
        elif len(new_string) > len(message):
            return (new_string[-len(message):] + "\n")
```
This returns the 'cleaned-up' version of every iteration runs:
```
vhfinmxkl atox kxgwxkxw tee hy maxlx hew vbiaxkl tl hulhexmx. px'ee atox mh kxteer lmxi ni hnk ztfx by px ptgm mh dxxi hnk fxlltzxl ltyx.

wigjonylm bupy lyhxylyx uff iz nbymy ifx wcjbylm um ivmifyny. qy'ff bupy ni lyuffs mnyj oj iol augy cz qy quhn ni eyyj iol gymmuaym muzy.
. . . 
. . .
computers have rendered all of these old ciphers as obsolete. we'll have to really step up our game if we want to keep our messages safe.
. . .
```