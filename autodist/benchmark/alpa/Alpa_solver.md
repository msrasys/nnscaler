# Alpa Solver Details

We have conducted a detailed test on the  solver in Alpa and have two conclusions.
1)  input and constraints limit the efficiency of the solver. 
2)  Alpa is unable to correctly solve problems with memory constraints. 

The table below shows the numbers of the Alpa solver under different micro batch sizes and GPT models in spmd. This table includes there parts, the number of free variables (the first two columns) , alpa solving times (the mid three columns) and autodist compile times (the last column). **Baseline time** represents the original solver time, **random** represents filling the array with random numbers, and **mem** represents adding memory constraint conditions. 

**1.3B**
|    |   num_nodes |   num_edges |   baseline time/s |   random time/s | mem time/s   |   autodist time/s |
|---:|------------:|------------:|------------------:|----------------:|:-------------|------------------:|
|  1 |        2637 |        4586 |           19.3861 |         19.832  | > 600        |              2.65 |
|  2 |        2472 |        4468 |           18.4109 |         34.7781 | > 600        |              3.96 |
|  4 |        2473 |        4470 |           21.6232 |         42.6273 | > 600        |              6    |
|  8 |        2473 |        4470 |           19.3299 |         25.2286 | > 600        |              9.78 |
| 16 |        2473 |        4470 |           19.34   |         43.1969 | > 600        |             14.91 |
| 32 |        2473 |        4470 |           20.1404 |         38.766  | > 600        |              9.29 |

**2.6B**
|    |   num_nodes |   num_edges |   baseline time/s |   random time/s | mem time/s   |   autodist time/s |
|---:|------------:|------------:|------------------:|----------------:|:-------------|------------------:|
|  1 |        3493 |        6090 |           27.2841 |         48.645  | > 600        |             17.57 |
|  2 |        3272 |        5932 |           27.0608 |         41.8054 | > 600        |             24.84 |
|  4 |        3272 |        5932 |           25.7738 |         67.6498 | > 600        |             36.19 |
|  8 |        3273 |        5934 |           29.1824 |         67.1933 | > 600        |             76.63 |
| 16 |        3273 |        5934 |           27.4334 |         33.0636 | > 600        |            115.04 |
| 32 |        3273 |        5934 |           30.3701 |         63.2393 | > 600        |             69.36 |

We can see from each row of the table that randomizing the input (**1.5~3x**) and adding memory constraint conditions (**>20x**) will increase the solving time of the solver, thus leading to the first conclusion stated above. Meanwhile, Autodist can quickly solve problems with memory constraints, achieving a maximum of **226x** faster solving efficiency (the first row at table 1.3B).  

For the second conclusion, we reduced the layers of GPT-3 1.3B from 24 to 12 under a 30GB memory constraint. Alpa was unable to find a solution (but Autodist is able to find one). We make experimental examples (shown in **Table\***) to state that Alpa solver with memory constraint is unreasonable.

**Table\***

The GPT model is 1.3B, we decrease the layer from 24 into 1, 5 and 12 respectively. Time ratio = mem time / baseline time.

|    |  baseline time/s |   mem time/s   |   time ratio |
|---:|-----------------:|---------------:|-------------:|
|  1 |    1.00          |        2.55    |         2.55 |
|  5 |    4.11          |        72.60   |        17.66 |
| 12 |    9.93          |  None solution |          --  |

There are two unreasonable aspects, the first aspect is that the time ratio increases exponentially as the gpt model increases. Second, Alpa solver will not search for a solution when the model becomes larger (although this solution certainly exists from the above statement).
