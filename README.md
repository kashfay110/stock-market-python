# A Quantitative Analysis of the Stocks of Different Sectors; Comparing the Ease of Prediction of One Sector from Another using Machine Learning Techniques.

## Methodology Requirements:
### Software, Programs, Languages etc. used:
- Python
- PyCharm IDE
- Google Collab
- Microsoft Excel
- Yahoo Finance

### Machine Learning Techniques used:
- LR (Linear Regression)
- SVR (Support Vector Regression)
- ANN (Articial Neural Network)

### Evaulation Metrics used:
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

## File Locations:
### Dataset:
- All datasets used are in the **dataset** directory, sorted by sector name. The top 10 stocks from each sector at the time were used.

### Results:
- Raw results for LR and SVR are in the **results/Sectors** directory, sorted by sector name. Results for 'FB' and 'AML' are present in the **results** directory.
- Raw results for ANN are in the **results-ann/Sectors** directory, sorted by sector name.
- Raw sorted results for everything, as well as some manipulation of results, are in the Excel workbook named **'All Results Final.xlsx'**. Each sector has its own sheet in the workbook.

### Python Scripts:
- The python script which was used to predict stock prices for 1,2,3,4,5,10,15 and 30 days into the future, using LR and SVR is **'svr_fb.py'**.
- The python script which was used to predict stock prices for 1,2,3,4,5,10,15 and 30 days into the future, using ANN is **'svr_ann_collab.ipynb'**.
- The python script used to merge all the results into one big Excel workbook is **'merge.py'**.

## Final Report and Discussion:
- This is all present in the PDF file named **'Complete Report.pdf'**. This contains the full report, including:
    - Introduction
    - Literature Review
    - Methodology
    - Results
    - Discussion
    - Conclusion.
- This is where the research questions are answered and the results are manipulated. 

