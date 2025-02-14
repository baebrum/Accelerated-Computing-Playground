Implement OpenMP locking to synchronize the parallel in-order insertion of Node structures into a bidirectional linked list.

Implement the bidirectional linked. This will lock each node and not scale with more than 1 thread. Using a bash script driver in a Makefile. Invoke a serial version of your code and measure the time taken to insert, in order, N = 2<sup>0</sup> to 2<sup>18</sup> Nodes with one thread.

Then, develop and invoke an OpenMP lock synchronized parallel version of your code and measure the time taken to insert, in order, N = 2<sup>0</sup> to 2<sup>18</sup> Nodes with more 1 threads. Experiment with different values of n as a power of 2.

Plot runtime vs. Node count, i.e., t (sec) vs. N,  and color the serial result line black. On the same graph, plot  t (sec) vs. N, for different thread count values of n, and choose a different color for each value of n. Include a legend in your plot and label each line with its respective value of n. Label your axes.

Based on your results, for which values of n do you observe speedup, and at which values of N?  Comment on your findings. In which circumstances is it beneficial to parallelize the insertion of data into a linked list?

Create a single PDF showing your single plot with several plot lines for a few thread count values of n, your comments, and your parallel code.

Verify the correctness of your algorithm by checking the intended number of Node structures were inserted, in ascending order, without deadlocking.