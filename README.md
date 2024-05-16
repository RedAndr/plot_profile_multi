# plot_profile_multi.py: A Python script for plotting and validating mercury model time series data.

**Instructions:**
1. Extract data files with directories from `DataObsMod.7z`.
2. Install dependences from `requirements.txt`: ```pip install requirements.txt```
3. Run the Script: ```python3 plot_profile_multi.py 2015 -coords -taylor -diurnal 3 3 -hist 8``` This example plots observed and modeled time series for the year 2015, including a Taylor diagram, diurnal profiles, and a histogram. The output will be saved in a PDF file named `2015.NAVEB93.F5B.dmu33th.pdf`.

```
usage: plot_profile_multi.py [-h] [--version] [-n] [-s] [-d] [--start START] [--end END] [-debias] [-nodaily] [-weekly] [-nomonthly] [-nostats] [-statplot] [-taylor] [-solar] [-biascor] [-diurnal [{1,2,3,4} ...]] [-coords] [-ymin YMIN] [-ymax YMAX] [-ydmin YDMIN] [-ydmax YDMAX]
                             [-tz TZ] [-hist [NBINS]] [-pngdpi [PNGDPI]] [-pngclr [PNGCLR]] [-pngopt]
                             year

Plot mercury time series

options:
  -h, --help            show this help message and exit

General options:
  --version             show program's version number and exit
  -n, --notrans         Disable transparency
  -s, --serial          Process in serial, not in parallel
  -d, --dry-run         Perform a dry run, simulating program execution without any actual changes

Time range options:
  year                  Target year
  --start START         Start month (default: 1)
  --end END             End month (default: 12)

Data processing options:
  -debias               Remove bias from the model data

Plotting options:
  -nodaily              Do not plot daily average
  -weekly               Plot weekly average
  -nomonthly            Do not plot monthly average
  -nostats              Do not write statistics on the plots
  -statplot             Plot statistics plots
  -taylor               Plot Taylor diagram
  -solar                Plot Solar diagram
  -biascor              Plot Bias-Correlatin diagram
  -diurnal [{1,2,3,4} ...]
                        Plot diurnal profiles, can accept monthly and hourly averages
  -coords               Add sites locations to the legend

Axis limits options:
  -ymin YMIN            Y-axis min for time series
  -ymax YMAX            Y-axis max for time series
  -ydmin YDMIN          Y-axis min for diurnal plot
  -ydmax YDMAX          Y-axis max for diurnal plot

Output options:
  -tz TZ                Local time zone (default: 0)
  -hist [NBINS]         Plot histogram with 'NBINS' bins, if 'NBINS' is not specified default is 16
  -pngdpi [PNGDPI]      Create png files with 'PNGDPI' resolution and collect them into 'pptx' file, default: 100
  -pngclr [PNGCLR]      Number of colors for png, default: 64
  -pngopt               Optimize png images for ppt files
```
