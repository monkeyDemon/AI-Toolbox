# Danbooru-grabber

_Upcoming update_: As some users pointed out, Danbooru limits the number of tags up to 2. 

_Update 2015/12/07_: Now, you can download all the pictures with certain tags. Simply type in a very large number like 1000. It will automatically stop when all pictures are downloaded. Also you will be able to download more than 100 pictures at once now.

**__Currently you would have to open the `grabber.py` and use text editor to change line 10 before using__**
#### About
This is a simple Python script that downloads pictures from [Danbooru](http://danbooru.donmai.us/). It downloads 50 pictures with tags pre-specified by users and saves the pictures to the folder named after the tags users input. **WARNING: You may end up downloading pictures that are NSFW depending on the tags you use. So use it wisely.**

#### Prerequisites
- First you need to [have Python installed](https://www.python.org/downloads/). This script is written in Python 2.7.10. It hasn't been tested under Python 3 yet.
- You will also need a Python modules called [requests](http://docs.python-requests.org/en/latest/). 

#### How do I run this thing (on Mac)?
- Open the terminal. 
- Navigate to the folder where you downloaded this script. For example, if you downloaded this script to `/Users/Nico`, then you can type in `cd /Users/Nico` and it will take you there.
- Type in `python grabber.py` and follow the instruction.
- Enjoy!
