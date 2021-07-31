import os,glob
import smtplib
from pandas import read_csv

progressBars = ['-','\\','|','/']

def loadcsv(f):
    # load csv file (faster than np.loadtxt())
    return read_csv(f).values

def showProgress(currSteps, totSteps):
    # report program progress
    # other option: tqdm
    print(' %c %.2f %%\r'%(progressBars[currSteps%4],100.*currSteps/totSteps),end='')

class emailHandler:
    # email notifier of running process
    def __init__(self, emailFrom, emailPw, emailTo):
        self.emailFrom = emailFrom
        self.emailPw = emailPw
        self.emailTo = emailTo
        self.connectServer()

    def connectServer(self):
        self.server = smtplib.SMTP('smtp.gmail.com:587')
        self.server.ehlo()
        self.server.starttls()
        self.server.login(self.emailFrom,self.emailPw)

    def setEmailTitle(self, emailTitle):
        self.emailTitle = emailTitle
        try:
            self.server.sendmail(self.emailFrom,self.emailTo,'\r\n'.join([
                'From: %s'%self.emailFrom,
                'To: %s'%self.emailTo,
                'Subject: (Program email) %s'%self.emailTitle,
                '',
                'process starts'
            ]))
        except:
            self.connectServer()
            self.setEmailTitle(emailTitle)

    def sendEmail(self, emailContent):
        try:
            self.server.sendmail(self.emailFrom,self.emailTo,'\r\n'.join([
                'From: %s'%self.emailFrom,
                'To: %s'%self.emailTo,
                'Subject: Re: (Program email) %s'%self.emailTitle,
                '',
                emailContent
            ]))
        except:
            self.connectServer()
            self.sendEmail(emailContent)

    def quitEmail(self):
        try:
            self.server.sendmail(self.emailFrom,self.emailTo,'\r\n'.join([
                'From: %s'%self.emailFrom,
                'To: %s'%self.emailTo,
                'Subject: (Program email) program completes'
            ]))
            self.server.quit()
        except:
            self.connectServer()
            self.quitEmail()
