% Author Hafsa Maryam and Tania Panayiotou.

% y are the several prediction from one test pattern
% t is our theshold  (similar to the quantile values)
function MC_estimate= MC_estimate(y,t) 
t=t;

[f,x] = ecdf(y); % create the cdf from the data
                 % f is the cdf output
                 % x are the values (i.e., bit-rate in our case over which the cdf is computed)
ecdf(y) 
index_estimates=find(f<=t); % find the indexes of all estimates for which their is a probability of less that t.

mc_estimates=x(length(index_estimates));
MC_estimate= max(mc_estimates); 





