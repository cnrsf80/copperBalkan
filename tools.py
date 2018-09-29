import random
import shutil
import os

def create_html(name="index.html",code="",url_base=""):
    file=open("./saved/"+name+".html","w")
    code=code.replace("\n","<br>")
    file.write(code)
    file.close()
    if len(url_base)>0:
        return url_base+"/"+name+".html"
    else:
        return ""


def tirage(str):
    return random.choice(str)


def clear_dir():
    shutil.rmtree("./saved")
    os.mkdir("./saved")