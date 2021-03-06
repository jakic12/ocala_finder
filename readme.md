# How does it work

If you want to generate your own timelapse from `ocala` images  
  
**like this one**
![Screenshot_20210307_002322](https://user-images.githubusercontent.com/37750012/110224381-70c79e00-7edb-11eb-8dec-f09ed9ecd68b.png)  

place the file `ocalaFinder.py` into the folder with your images and call  

```
./ocalaFinder.py
```  
  
### Note

If the program finds the two circles, it will cache their positions in the file  
`<filename>_bestBoiPosition.txt`  
So the next time you will run the program, it will ignore the cached files. If you want to force the search to all files,  
  
run
```
rm *_bestBoiPosition.txt
```  

# Prerequisites
* `python3` is required to run
* All images must be in the folder next to the script
* All images must use the extension `.jpg` or `.png` (search the code for `@%^magic_string@%^` if you need to alter it)

# Post-processing
Use the following script to remove the `xx_adj_` prefix from your images:  
```
rename -n 's/\d*_adj_(.*)/$1/' *
```  
if you're satisfied with the result, remove the simulation flag (`-n`)  
(`rename` needs to be installed for this)

