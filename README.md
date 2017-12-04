# CSC592-ML-FinalProject

# csc592finalproject

// setup for the first time, do this in the directory where you want to store the files
git clone https://github.com/mbellows/CSC592-ML-FinalProject.git

-------------
// When doing all these commands in the terminal, you must be in the path you specified for the project

// see what is changed, gives you an idea what you are about to do<br/>
git fetch<br/>
git status 

// add all changed files to your 'stage' (stuff that will be committed)<br/>
git add --all 

// commit your changes (local only)<br/>
git commit -m "some helpful description of what you did"

// share your changes with everyone else<br/>
git push origin

// get other people's changes<br/>
git fetch<br/>
git pull

------------

// dealing with conflicts (two people change the same piece of the same file independently)
// step 1: open the file with conflicts and look for sections that look like this:<br/>

<<<<< head<br/>
stuff you change<br/>
======<br/>
stuff they changed<br/>
>>>>>> their commit name<br/>

// step 2: combine / resolve the changes so both people are happy<br/>
// step 3: tell git the changes are resolved<br/>
git add <file name><br/>

// commit and push as per usual

-----------

// throw away your local changes (for example, so you can pull and take remote)

git add --all<br/>
git reset --hard<br/>

then...git pull
