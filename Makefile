# Description: Makefile for template processing
# Read the README.md for more information on how to use this Makefile.
# Or run `make` for online help.

#### ENABLE LOGGING FIRST
# USER-VARIABLE: LOGGING_LEVEL
# Defines the logging level for the Makefile.

# Load make logging function library
include cookbook/log.mk


# USER-VARIABLE: CONFIG_LOCAL_MAKE
# Defines the name of the local configuration file to include.
#
# This file is used to override default settings and provide local configuration. If a
# file with this name exists in the current directory, it will be included. If the file
# does not exist, it will be silently ignored. Never add the file called config.local.mk
# to the repository! If you have stored config files in the repository set the
# CONFIG_LOCAL_MAKE variable to a different name.
CONFIG_LOCAL_MAKE ?= config.local.mk

# Load local config if it exists (ignore silently if it does not exists)
-include $(CONFIG_LOCAL_MAKE)


# Report logging level after processing local configurations
  $(call log.info, LOGGING_LEVEL)


#: Show help message
help::
	@echo "Makefile for myprocessing processing"
	@echo "Usage: make <target>"
	@echo "Targets:"

# Set special .DEFAULT_GOAL variable to help target
.DEFAULT_GOAL := help
.PHONY: help


# Set shared make options
include cookbook/make_settings.mk

# If you need to use a different shell than /bin/dash, overwrite it here.
# SHELL := /bin/bash



# SETUP SETTINGS AND TARGETS
include cookbook/setup.mk
include cookbook/setup_python.mk
# for asw tool configuration if needed
# include cookbook/setup_aws.mk
# for myprocessing configuration, adapt to your needs
include cookbook/setup_myprocessing.mk

# Load newspaper list configuration and processing rules
include cookbook/newspaper_list.mk


# SETUP PATHS
# include all path makefile snippets for s3 collection directories that you need
include cookbook/paths_rebuilt.mk
include cookbook/paths_myprocessing.mk


# MAIN TARGETS
include cookbook/main_targets.mk


# SYNCHRONIZATION TARGETS
include cookbook/sync.mk
include cookbook/sync_rebuilt.mk
include cookbook/sync_myprocessing.mk

include cookbook/clean.mk


# PROCESSING TARGETS
include cookbook/processing.mk
include cookbook/processing_myprocessing.mk


# FUNCTION
include cookbook/local_to_s3.mk


# FURTHER ADDONS
