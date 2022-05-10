import os

class Config(object):
    SECRET_KEY =  os.environ.get('SECRET_KEY') or "secret_string"
    #en special key - signature key - alt der bliver sendt til 
    #server ikke bliver altered eller hacked 
    #MONGODB_SETTINGS = { 'db' : 'UTA_Enrollment'} #UTA_enrollment er navnet på databasen
    