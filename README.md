## Overview
manual for reproducing results

Disclaimer:
This code is provided for research use only. The authors do not have
any responsibility for the results generated by the code or the 
accuracy thereof. For any other use, please contact the Authors.

## Instructions to run and test
paper1_refactor.py


idealsearch=2 extreme point search

idealsearch=5 corner 1 search + all evaluation

idealsearch=4 corner 1 search + selective evaluation 

idealsearch=6 corner 1 search + selective evaluation + Silhouette analysis

idealsearch=3 corner 2 search + selective evaluation + Silhouette analysis

How to generate results:

(1) raw process:
trainy_summary2csv(resultfolder, resultconver)

input: resultfolder(where all results are stored)
       resultconver(where results processing outcome is stored)

(2) next step

In the folder named by resultconver (e.g. paper1_convert)
move  matlab script: randsum_4hv.m to this folder 
and run it the file 'hvcompare_sig.csv' file will be generated
it is latex compatible file for the paper table 2-4

(3) figure plots:
(3-1) convergence plot: hvconverge_averageplot
similar as (1) provide result folder and the second input is not used
		output is in  process_plot folder under the result folder
provide problem json file p/resconvert_plot3.json, which problem to process

(3-2) final results plot in plot_for_paper
as above provide which problem to solve in resconvert_plot3.json
results are found in process_plot folder under the result folder

as for the init plot, uncomment the init script
run as above 



(4) demo figures
degenerate front: plot_run in paper1_refactor.py
corner demo: demo_plot2 in paper1_resconvert.py