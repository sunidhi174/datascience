---
title: Julia In A Nutshell
tags: [tutorial]
header:
excerpt: "A gentle introduction to Julia Language"
---
---


The official website states following about the Julia lang:

1. Julia is fast!
2. Dynamic
3. Optionally type
4. General
5. Easy to use
6. Open source

In-short Julia is a high-level scientific programming langauge designed for performance and compete with MATLAB and C++. With over 1900 packages Julia is aiming to replace Python for Data Science research. 

In these tutorials I've used snippets of codes I found at [juliaBox.](https://github.com/JuliaComputing/JuliaBoxTutorials) They have great notebooks for every topic.

In these series of tutorials we'll cover the concepts of Julia and go on to build a simple Machine Learning Model

## Topics Covered:
1. How to print
2. How to assign variable
3. Math Syntax
4. How to get strings
5. String Manipulation
6. String concatentation
7. Data Structures
    1. Tuples
    2. Dict
    3. Arrays
8. Loops

### How to print:

in Julia we using ```println()``` function to print to the standard I/O let us start with the customary hello world


```julia
println("Hello, World!")
```

    Hello, World!


### How to assign variable

In Julia, like Python we use equal sign to assign a value to a variable.

The variable can be of any datatype Being a dynamic programming language it will indentify the datatype automatically

We can check the type of a variable using ```typeof()``` function


```julia
var1 = 200
typeof(var1)
```




    Int64




```julia
var2 = "Julia"
typeof(var2)
```




    String



In Julia we can use emojis to make the documents fun to read. 

```julia
#\:smi + <tab> --> select with down arrow + <enter> ---> <tab> + <enter> to complete
```

### Syntax for math


```julia
sum = 3 + 7
```




    10




```julia
difference = 10 - 3
```




    7




```julia
product = 20 * 5
```




    100




```julia
quotient = 100 / 10
```




    10.0




```julia
power = 10 ^ 2
```




    100




```julia
modulus = 101 % 2
```




    1



Alright this should get you fimilar with basic Julia. Let's crank it up and some basic String Manipulation.

### How to get Strings in Julia.

Julia provides two ways to enclose strings:
1. " "
2. """ """

There are a couple functional differences between strings enclosed in single and triple quotes.
One difference is that, in the latter case, you can use quotation marks within your string.

Note: ' ' are used for a character and not a string


```julia
"Here, we get an "error" because it's ambiguous where this string ends "
```


    syntax: cannot juxtapose string literal

    



```julia
"""Look, Mom, no "errors"!!! """
```




    "Look, Mom, no \"errors\"!!! "



### String manipulation

Like Python's '%' character we can insert variables in strings using '$' as a prefix to the variable name


```julia
name = "Jane"
num_fingers = 10
num_toes = 10
```




    10




```julia
println("Hello, my name is $name.")
println("I have $num_fingers fingers and $num_toes toes.")
```

    Hello, my name is Jane.
    I have 10 fingers and 10 toes.


### String Concatentation

Below are two ways we can concatenate strings!

The first way is to use the string() function.


```julia
s3 = "This is string 1"
s4 = "This is what string 2 looks like"
```




    "This is what string 2 looks like"




```julia
string(s3, s4)
```




    "This is string 1This is what string 2 looks like"



The second way is to use  '\*' operator on string 


```julia
s3*s4
```




    "This is string 1This is what string 2 looks like"



### Data Structures

Julia has three primary data structures to handle data

1. tuple
2. Dictionary
3. Array

#### Tuples

Tuples are an immutable (meaning we cannot change it) , ordered data type (meaning it can be indexed). To initiate a tuple we use ```( )```

oh while we are at it... Julia is 1 indexed.

Yes, Julia is 1-based indexing, not 0-based like Python. Wars are fought over lesser issues. I have a friend with the wisdom of Solomon who proposes settling this once and for all with ½ 😃


```julia
t1 = (1,2,3)
```




    (1, 2, 3)



Since tuples are ordered we can index them using ```[]```


```julia
t1[2]
```




    2



since tuples are immutable we can't update an element


```julia
t1[1] == 2
```




    false



As you might guess, `NamedTuple`s are just like `Tuple`s except that each element additionally has a name! They have a special syntax using `=` inside a tuple:

```julia
(name1 = item1, name2 = item2, ...)
```


```julia
myfavoriteanimals = (bird = "penguins", mammal = "cats", marsupial = "sugargliders")
```




    (bird = "penguins", mammal = "cats", marsupial = "sugargliders")



Like regular `Tuples`, `NamedTuples` are ordered, so we can retrieve their elements via indexing:


```julia
myfavoriteanimals[1]
```




    "penguins"



They also add the special ability to access values by their name:


```julia
myfavoriteanimals.bird
```




    "penguins"



#### Dictionaries

If we have sets of data related to one another, we may choose to store that data in a dictionary. We can create a dictionary using the `Dict()` function, which we can initialize as an empty dictionary or one storing key, value pairs.

Syntax:
```julia
Dict(key1 => value1, key2 => value2, ...)```

A good example is a contacts list, where we associate names with phone numbers.


```julia
myphonebook = Dict("Jenny" => "867-5309", "Ghostbusters" => "555-2368")
```




    Dict{String,String} with 2 entries:
      "Jenny"        => "867-5309"
      "Ghostbusters" => "555-2368"




```julia
myphonebook["Jenny"]
```




    "867-5309"




```julia
myphonebook["Kramer"] = "555-FILK"
```




    "555-FILK"



We can delete elements from the dictionary using ```pop!``` (note the ! mark it's part of the syntax)


```julia
myphonebook
```




    Dict{String,String} with 3 entries:
      "Jenny"        => "867-5309"
      "Kramer"       => "555-FILK"
      "Ghostbusters" => "555-2368"




```julia
pop!(myphonebook, "Kramer")
```




    "555-FILK"




```julia
myphonebook
```




    Dict{String,String} with 2 entries:
      "Jenny"        => "867-5309"
      "Ghostbusters" => "555-2368"



#### Arrays

Unlike tuples, arrays are mutable. Unlike dictionaries, arrays contain ordered collections. <br>
We can create an array by enclosing this collection in `[ ]`.

Syntax: <br>
```julia
[item1, item2, ...]```

For example we can create an array of heights of people at Factspan in centimeter:


```julia
height = [168, 156, 171, 180, 177]
```




    5-element Array{Int64,1}:
     168
     156
     171
     180
     177



The `1` in `Array{Int64,1}` means this is a one dimensional vector.  An `Array{Int64,2}` would be a 2d matrix, etc.  The `Int64` is the type of each element.


```julia
names = ["Ronak", "Shweta", "Divyani", "Raghu", "Nitin"]
```




    5-element Array{String,1}:
     "Ronak"  
     "Shweta" 
     "Divyani"
     "Raghu"  
     "Nitin"  



names contains only ```String``` variables. In Julia, an array can contain multiple data types:  


```julia
mix = [12, 14, 15, "lorem", "ipsum"]
```




    5-element Array{Any,1}:
     12       
     14       
     15       
       "lorem"
       "ipsum"



We can add variable to an array using ```push!()``` and remove the variables using ```pop!()```

```push!()``` adds data at the end of the of array

```pop!()``` removes data from the end of the array


```julia
height
```




    5-element Array{Int64,1}:
     168
     156
     171
     180
     177




```julia
push!(height, 200)
```




    6-element Array{Int64,1}:
     168
     156
     171
     180
     177
     200




```julia
height
```




    6-element Array{Int64,1}:
     168
     156
     171
     180
     177
     200



So far we are working with 1D arrays. Julia also supports 2D Arrays. 

Let us create an array of arrays which are inherently different that 2D arrays


```julia
dim2 = [[1,2,3],[7,8,9],[11,12,13]]
```




    3-element Array{Array{Int64,1},1}:
     [1, 2, 3]   
     [7, 8, 9]   
     [11, 12, 13]




```julia
dim2[2][3]
```




    9



This will give an error


```julia
dim2[2,3]
```


    BoundsError: attempt to access 3-element Array{Array{Int64,1},1} at index [2, 3]

    

    Stacktrace:

     [1] getindex(::Array{Array{Int64,1},1}, ::Int64, ::Int64) at ./array.jl:732

     [2] top-level scope at In[39]:1


Let's create a 2D array using ```rand()``` function


```julia
arr2 = rand(3,3)
```




    3×3 Array{Float64,2}:
     0.785026  0.358534  0.392247
     0.229277  0.796046  0.400125
     0.791307  0.303603  0.823874




```julia
arr3 = rand(3,3,3)
```




    3×3×3 Array{Float64,3}:
    [:, :, 1] =
     0.287473  0.12579   0.623518
     0.390762  0.281463  0.283141
     0.877599  0.547904  0.533407
    
    [:, :, 2] =
     0.121765   0.0701316  0.164792 
     0.0685404  0.986686   0.0988056
     0.404107   0.320219   0.0401538
    
    [:, :, 3] =
     0.946193  0.506131   0.745722
     0.947387  0.24257    0.735788
     0.233367  0.0682487  0.993635




```julia
arr2[2,3]
```




    0.4001254300230308



#### Loops

Like all programming languages Julia has two loops:

1. ```while```
2. ```for```

the syntax of a ```while``` loops like 

```julia
while *condition*
    * loop body*
end
```

For example:


```julia
n = 0
while n < 10
    n += 1
    println(n)
end
n
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10





    10



The syntax for a `for` loop is

```julia
for *var* in *loop iterable*
    *loop body*
end
```

We could use a for loop to generate the same results as either of the examples above:


```julia
myfriends = ["Ted", "Robyn", "Barney", "Lily", "Marshall"]

for friend in myfriends
    println("Hi $friend, it's great to see you!")
end
```

    Hi Ted, it's great to see you!
    Hi Robyn, it's great to see you!
    Hi Barney, it's great to see you!
    Hi Lily, it's great to see you!
    Hi Marshall, it's great to see you!


A for loop can be used to create arrays Julia has some syntatic beauty:


```julia
m, n = 5, 5
A = fill(0, (m, n))

for i in 1:m, j in 1:n
    A[i,j] = i+j
end
```


```julia
A
```




    5×5 Array{Int64,2}:
     2  3  4  5   6
     3  4  5  6   7
     4  5  6  7   8
     5  6  7  8   9
     6  7  8  9  10



The Julia way, Like python, to construct an array is array comprehension


```julia
B = [i+j for i in 1:m, j in 1:n]
```




    5×5 Array{Int64,2}:
     2  3  4  5   6
     3  4  5  6   7
     4  5  6  7   8
     5  6  7  8   9
     6  7  8  9  10




```julia

```
