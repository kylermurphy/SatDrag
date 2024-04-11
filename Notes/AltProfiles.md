# Altitude Profiles

Test the best way to incorporate altitude into a single satellite density dataset. 

Using MSIS (pymsis) modelled density profiles create a density profile at the location of the GRACE satellites. This will be done by _scaling_ the modeled density profile to the density the GRACE satellites.

Two methods will be tested: 

1. Determine the ratio between the density at the altitude of GRACE B. Use this ratio to shift the MSIS density profile to match the observed density at the altitude of GRACE B. 
1. Calculate the scale height from the MSIS model and apply this scale height to GRACE B data to derive an altitude profile. 

Both methods will be compared to CHAMP and GOCE data to determine the best method for deriving a density altitude profile from observations.

Density profiles will be drived for time periods when GRACE is in close proximity to either CHAMP or GOCE. The figure below is a histogram of the difference in longitude between GOCE and GRACE at conjugate points in time. Due to the difference in altitude of the spacecraft and precession of the orbits, this difference varies. However, there are a large number of points when the seperation is within 5$^{\circ}$.

![GOCE/GRACE delta longitude](GOCE_hist.png)

During periods when the spacecraft are close in longitude their still may be a large displacement in latitude. However, each orbit the spacecraft will come to similar latitudes within at most 90 minutes of eachother (roughly the period of the orbits). These times, when the spacecraft orbits are close in local time (longitude) and pass through similar latitudes, can be used to derive altitude profiles at GRACE and compare to the altitude at GOCE (and CHAMP).

It is important to note that as the spacecraft move along their orbit they can be very close in longitude at one point in time but far away at another point in time. To ensure the orbits are always close we can calculate the difference in longitude between the two spacecraft when they are at the same latitdue. 

An example of a point when GOCE and GRACE are close is shown below. Here at 2013-10-20 03:10:10 GOCE is at 74.6$^{/circ} longitude and 58.6$^{/circ} latitude. A few minutes later 2013-10-20 03:15:00 GRACE is at 74.1^{/circ} and 58.5^{/circ} latitude. 

![GOCE (purple), GRACE (teal) @ 03:17](GOCE_GRACE_ex.png)

For this study we derive altitude profiles at GRACE for periods of time when the GRACE satellite is within 5$^{\circ}$ longitude 1$^{\circ}$ latitude of either CHAMP or GOCE within a single GRACE orbit (90 minutes). This yeilds 69,216 conjunctions between GRACE and GOCE, and 202,046 conjunctions between GRACE and CHAMP.  

