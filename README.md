# Bike Sharing Demand analysis

We must predict the total count of bikes rented during each hour covered by the test set, using only information available prior to the rental period.

### Data

1. datetime - hourly date + timestamp  
2. season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
3. holiday - whether the day is considered a holiday
4. workingday - whether the day is neither a weekend nor holiday
5. weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
6. temp - temperature in Celsius
7. atemp - "feels like" temperature in Celsius
8. humidity - relative humidity
9. windspeed - wind speed
10. casual - number of non-registered user rentals initiated
11. registered - number of registered user rentals initiated
12. count - number of total rentals

### Key inferences

1. Use all information stored in date field. Feeds must include:
    * hour
    * month
    * day_of_weed (1-7)
    * year
    
2. It is a good idea to use log transform (```np.log1p```, ```np.expm1```)
for 'count' column. The data is skewed to the left so log transformation
   makes it more 'normal' thus improves overall score from 0.5 to 0.42
   (the most valuable improvement)
   
### Model

1. tuned LGB (0.4)
2. tuned XGB (0.41)