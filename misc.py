import sys # for writing in the same line via sys.stdout
import os

#import own logger function
from logger import Logger
SELF="misc.py"
#instantiate logger class
logger = Logger(SELF)


def directory_creator(dir_name):
  #check if directory exists
  if not os.path.isdir(dir_name):
    logger.log(str("Creating output directory {}...".format(dir_name)))
    try:
      os.mkdir(dir_name)
      logger.log(str("Creating output directory {}...".format(dir_name)),logger.OK)
    except:
      logger.log(str("Creating output directory {}...".format(dir_name)),logger.FAIL)
      logger.log_simple(str("Could not create output directory {}".format(dir_name)))

