#!bin/bash
rm figures/wavelength/portA/*
rm figures/wavelength/portB/*
rm figures/wavelength/*.*
rm figures/*.*
rm figures/freequency/*.*
rm figures/freequency/portA/*
rm figures/freequency/portB/*
python mm_gnlse_2D.py
