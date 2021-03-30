#!/usr/bin/python3

import os
import sys


RED   = "\033[0;31m" 
BRED  = "\033[1;31m" 
BLUE  = "\033[0;34m"
BBLUE  = "\033[1;34m"
YELLOW = "\033[0;33m"
BYELLOW = "\033[1;33m"
CYAN  = "\033[0;36m"
BCYAN  = "\033[1;36m"
GREEN = "\033[0;32m"
BGREEN = "\033[1;32m"
RESET = "\033[0;0m"
ITALIC = "\033[;3m"
BOLD    = "\033[;1m"
REVERSE = "\033[;7m"



class Logger(object):
  '''
  This class is devoted to handle our special logging functions/features
  '''
  def __init__(self, called_from, **kwargs):
    '''
    Only the called_from argument is defined here. It is useful when the same 
    logger class is instantiated from different programs. 
    **kwargs are for future use
    '''
    self.called_from = called_from
    l=len(self.called_from)
    pad=" "
    if l < 20:
      padding=(20-l)*pad
    self.called_from+=padding

    self.logfile = kwargs.get('logfile',None)
    

    # print(self.called_from)

    ## FOR COLORING
    self.OK="OK"
    self.FAIL="FAIL"
    self.WARN="WARN"
    self.FOUND="FOUND"
    self.SKIP="SKIP"

    self.STATUS = {
          self.OK:BGREEN, 
          self.FAIL:BRED, 
          self.WARN:BYELLOW, 
          self.FOUND:GREEN,
          self.SKIP:BCYAN
          }
    self.STATUS_MSG = {
                  self.OK:    self.STATUS[self.OK]    + "[DONE]"+RESET,
                  self.FAIL:  self.STATUS[self.FAIL]  + "[FAIL]"+RESET,
                  self.WARN:  self.STATUS[self.WARN]  + "[WARN]"+RESET,
                  self.FOUND: self.STATUS[self.FOUND] + "[FOUND]"+RESET,
                  self.SKIP:  self.STATUS[self.SKIP]  + "[SKIP]"+RESET
    }

    self.TITLE_OPEN="TITLE_OPEN"
    self.TITLE_CLOSE="TITLE_CLOSE"
    self.TITLE_NONE="TITLE_NONE"
    # self.TITLE = {
    #               self.TITLE_OPEN = 1,
    #               self.TITLE_CLOSE = 2,
    #               self.TITLE_NONE = 0,
    # }
    ## -------------

    #get the terminal dimension the program is running from
    self.cols,self.rows=os.get_terminal_size(0)
    self.last_percent = 0

  def log(self, msg, status=None):
    prefix = str("{}{}{} -| ".format(ITALIC+BOLD,self.called_from,RESET))
    #line = str("{}{}{} -| {}".format(ITALIC+BOLD,self.called_from,RESET, msg)) # the full line of message
    white_space = " " #one white space for scaling
    if status is None:
      sys.stdout.write(str("{}\r".format(prefix+msg)))
    else:
      padding = self.cols - len(prefix+msg) - len(self.STATUS_MSG[status])+4 #+4 because of the 2 coloring chars and the 2 brackets
      if status not in self.STATUS_MSG:
        #this case, the class is not used properly, we do exit from
        sys.stdout.write(str("{}Status {} is unknown{}".format(RED, status, RESET)))
        sys.stdout.flush()
        exit(-1)

      if(padding > 0): #we have space for padding
        white_space=padding*white_space   
        sys.stdout.write(str("{}{}{}{}\n".format(prefix, #the prefix (called_from)
                                                msg,
                                                white_space, #padding
                                                self.STATUS_MSG[status]))) #closing tag 
        if(self.logfile is not None):
          
          self.f = open(self.logfile, "a")
          self.f.write(str("{}{}{}{}\n\n".format(prefix, #the prefix (called_from)
                                                msg,
                                                white_space, #padding
                                                self.STATUS_MSG[status]))) #closing tag
          self.f.close()
      
      else: #no space for padding, use a Tab and don't care :D
        sys.stdout.write(str("{}\t{}\n".format(prefix+msg, #the actual content
                                              self.STATUS_MSG[status]))) #closing tag
        if(self.logfile is not None):
          self.f = open(self.logfile, "a")
          self.f.write(str("{}\t{}\n\n".format(prefix+msg, #the actual content
                                              self.STATUS_MSG[status]))) #closing tag
          self.f.close()
      sys.stdout.flush()


  def log_simple(self, msg, title="TITLE_NONE"):
    '''
    Prefixed and postfixed message printer for titles, and regular printer for
    normal status messages
    Enum TITLE_NONE - default: simple printing
    Enum TITLE_OPEN -  "+------------- TITLE -------------+"
    Enum TITLE_CLOSE - "+=========== END TITLE ===========+"
    '''
    if (title == self.TITLE_NONE): # TITLE_NONE, i.e., normal printing
      line = str("{}{}{} -| {}".format(ITALIC+BOLD,self.called_from,RESET, msg)) # the full line of message
    else:
      padding = int(((self.cols - len(msg))/2 - len(self.called_from) - 3)-4) #padding for prefix and postfix + 2 chars for the '+' signs, plus two white spaces" " 
      
      if(title == self.TITLE_OPEN): #TITLE_OPEN
        prefix = str("+{} ".format(padding*"-"))
        postfix = str(" {}+".format(padding*"-"))
        line = str("{}{}{} -| {}{}{}".format(ITALIC+BOLD,self.called_from,RESET, prefix, msg, postfix)) # the full line of message

      else: #TITLE_CLOSE        
        prefix = str("+{} ".format(padding*"="))
        postfix = str(" {}+".format(padding*"="))
        line = str("{}{}{} -| {}{}{}".format(ITALIC+BOLD,self.called_from,RESET, prefix, msg, postfix)) # the full line of message

    print(line)
    if(self.logfile is not None):
      self.f = open(self.logfile, "a")
      self.f.write(line + str("\n"))
      self.f.close()


  def calculateRemainingPercentage(self, current, n, task):
    '''
    This function serves as a progress tracking feature for longer tasks
    @params:
    Int current - actual state
    Int n - all states
    Str task - some label to print
    '''
    percent = (int((current / float(n)) * 100))
    #to only update a row if there is any new value to show
    if(percent > self.last_percent):
      sys.stdout.write(str("{}{}{} -| {}...{}%\r".format(ITALIC+BOLD,self.called_from,RESET,task,percent)))
      sys.stdout.flush()  
      self.last_percent = percent
    
    #reset back to normal
    if(percent == 0 and self.last_percent == 100):
      #a new printout has just began
      #we reset last_percent to 0 to update again properly
      self.last_percent = 0
