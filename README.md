# population-initialization

Put together the population pyramid sampling for age with Kaplan-Meier curve sampling for date of death with a heap-based priority queue implementation to line all the agents up in date of death order.

## Performance

### Python Only

Commit 73965c431d7b9576edc862ac5ab2db7277a61f98
Population 1,000,000

|Phase|Wall Clock Time|
|-----|:-------------:|
|Build alias table (21 entries)|< 00:01s|
|Use DOB to draw for DOD|00:13s|
|Populate Priority Queue|00:02s|
|Iterate over 101 years, by day (maximum DOD)|00:21s|

### Numba

