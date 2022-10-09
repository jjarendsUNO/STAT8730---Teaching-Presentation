Presentation Code
================
Justin Arends
2022-10-13

# Performance Measures

-   With unbalanced data, overall accuracy is often inappropriate
-   True negative, true positive, weighted accuracy, G-mean, precision,
    recall, and F-measure

$True\ Negative (Acc^-) = \frac{TN}{TN + FP}$  
$True\ Positive (Acc^+)= \frac{TP}{FP + FN}$  
$G-mean = (Acc^- \times Acc^+)^{1/2}$  
$Weighted\ Accuracy = \beta Acc^- + (1-\beta)Acc^-$  
$Precision = \frac{TP}{TP+FN}$  
$Recall = \frac{TP}{TP+FN} = Acc^+$  
$F-Measure = \frac{2 \times Precision \times Recall}{Precision+Recall}$

# Modeling

## Oil Data

    ## Rows: 937 Columns: 50
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## dbl (50): X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, ...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

Custom Measures

    ## # A tibble: 2 × 2
    ##   X50       n
    ##   <fct> <int>
    ## 1 0       672
    ## 2 1        30

    ## # A tibble: 9 × 6
    ##   .metric           .estimator  mean     n  std_err .config             
    ##   <chr>             <chr>      <dbl> <int>    <dbl> <chr>               
    ## 1 accuracy          binary     0.962    25 0.00199  Preprocessor1_Model1
    ## 2 f_meas            binary     0.981    25 0.00104  Preprocessor1_Model1
    ## 3 f_meas_beta       binary     0.366    22 0.0209   Preprocessor1_Model1
    ## 4 precision         binary     0.964    25 0.00204  Preprocessor1_Model1
    ## 5 recall            binary     0.998    50 0.000374 Preprocessor1_Model1
    ## 6 recall_second_vec binary     0.212    25 0.0218   Preprocessor1_Model1
    ## 7 roc_auc           binary     0.897    25 0.0106   Preprocessor1_Model1
    ## 8 sens              binary     0.998    25 0.000534 Preprocessor1_Model1
    ## 9 specificity       binary     0.212    25 0.0218   Preprocessor1_Model1

    ## # A tibble: 10 × 4
    ##    .metric           .estimator .estimate .config             
    ##    <chr>             <chr>          <dbl> <chr>               
    ##  1 accuracy          binary        0.949  Preprocessor1_Model1
    ##  2 specificity       binary        0.0909 Preprocessor1_Model1
    ##  3 recall            binary        0.991  Preprocessor1_Model1
    ##  4 f_meas            binary        0.974  Preprocessor1_Model1
    ##  5 f_meas_beta       binary        0.143  Preprocessor1_Model1
    ##  6 sens              binary        0.991  Preprocessor1_Model1
    ##  7 recall            binary        0.991  Preprocessor1_Model1
    ##  8 recall_second_vec binary        0.0909 Preprocessor1_Model1
    ##  9 precision         binary        0.957  Preprocessor1_Model1
    ## 10 roc_auc           binary        0.908  Preprocessor1_Model1

## Mammography

    ## Rows: 11183 Columns: 7
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (1): X7
    ## dbl (6): X1, X2, X3, X4, X5, X6
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

### Training/Testing Splits

### Custom Function to pull back metrics

![](Justin-Presentation-Code_files/figure-gfm/Metrics%20Table-1.png)<!-- -->

    ## # A tibble: 7 × 7
    ##   model        Precision   TNR   TPR `G-Mean` `F-Measure` `Weighted Accuracy`
    ##   <chr>            <dbl> <dbl> <dbl>    <dbl>       <dbl>               <dbl>
    ## 1 SMOTE 100        0.735 0.993 0.568    0.751       0.641               0.781
    ## 2 SMOTE 200        0.73  0.993 0.598    0.771       0.658               0.796
    ## 3 Weighted 2:1     0.525 0.989 0.847    0.915       0.648               0.918
    ## 4 Weighted 3:1     0.54  0.989 0.850    0.917       0.661               0.920
    ## 5 Balanced .5      0.537 0.989 0.795    0.886       0.641               0.892
    ## 6 Balanced .6      0.540 0.989 0.8      0.889       0.645               0.895
    ## 7 Balanced .7      0.535 0.989 0.799    0.889       0.641               0.894
