

## R Code Used in the Examples - tsda 

This is an updated version of the code in [Time Series: A Data Analysis Approach
Using R](http://www.stat.pitt.edu/stoffer/tsda/) 


&#x2728; While the text was written under `astsa` version 1.9, the code below uses the most recent version, which has some additional capabilities.  You can install the latest version of the package from GitHub:

```r
install.packages("remotes")     # only need to do this once 
remotes::install_github("nickpoison/astsa")
```


### An intro to `astsa` capabilities can be found at  [FUN WITH ASTSA](https://github.com/nickpoison/astsa/blob/master/fun_with_astsa/fun_with_astsa.md)

---

>  Note: when you are in a code block below, you can copy the contents of the block by moving your mouse to the upper right corner and clicking on the copy icon ( &#128203; ).

-----
------ 

### Table of Contents
  
  * [Chapter 1](#chapter-1)
  * [Chapter 2](#chapter-2)
  * [Chapter 3](#chapter-3)
  * [Chapter 4](#chapter-4)
  * [Chapter 5](#chapter-5)
  * [Chapter 6](#chapter-6)
  * [Chapter 7](#chapter-7)
  * [Chapter 8](#chapter-8)
 
---

## Chapter 1

Example 1.1 

```r
par(mfrow=2:1)
tsplot(jj, ylab="QEPS", type="o", col=4, main="Johnson & Johnson Quarterly Earnings")
tsplot(log(jj), ylab="log(QEPS)", type="o", col=4)
```

Example 1.2  

```r
tsplot(cbind(gtemp_land,gtemp_ocean), spaghetti=TRUE, col = astsa.col(c(2,4), .5), 
        lwd=2, type="o", pch=20, ylab="Temperature Deviations", main="Global Warming")
legend("topleft", col=c(2,4), lty=1, lwd=2, pch=20,  bg="white",
        legend=c("Land Surface", "Sea Surface")
``` 

Example 1.3  

```r
library(xts)   # install.packages("xts") if you don't have it already 
djia_return = diff(log(djia$Close))[-1]
par(mfrow=2:1)
plot(djia$Close, col=4)
plot(djia_return, col=4)

tsplot(diff(log(gdp)), type="o", col=4) # using diff log
points(diff(gdp)/lag(gdp,-1), pch="+", col=2) # actual return
``` 

Example 1.4  

```r
par(mfrow = c(2,1))
tsplot(soi, ylab="", xlab="", main="Southern Oscillation Index", col=4)
text(1970, .91, "COOL", col=5)
text(1970,-.91, "WARM", col=6)
tsplot(rec, ylab="", main="Recruitment", col=4) 
```

Example 1.5  

```r
tsplot(Hare, col = astsa.col(2, .5), lwd=2, type="o", pch=0, ylab=expression(Number~~~(""%*% 1000)))
lines(Lynx, col=astsa.col(4, .5), lwd=2, type="o", pch=2)
legend("topright", col=culer, lty=1, lwd=2, pch=c(0,2), legend=c("Hare", "Lynx"), bty="n")
```

Example 1.6
```r
par(mfrow=c(3,1))
u = rep(c(rep(.6,16), rep(-.6,16)), 4) # stimulus signal
tsplot(fmri1[,4], ylab="BOLD", xlab="", main="Cortex", col=4, ylim=c(-.6,.6), lwd=2)
 lines(fmri1[,5], col=6, lwd=2)
 lines(u, type="s")
tsplot(fmri1[,6], ylab="BOLD", xlab="", main="Thalamus", col=4, ylim=c(-.6,.6), lwd=2)
 lines(fmri1[,7], col=6, lwd=2)
 lines(u, type="s")
tsplot(fmri1[,8], ylab="BOLD", xlab="", main="Cerebellum", col=4, ylim=c(-.6,.6), lwd=2)
 lines(fmri1[,9], col=6, lwd=2)
 lines(u, type="s")
mtext("Time (1 pt = 2 sec)", side=1, line=1.75)
```

Example 1.7 - 1.8

```r
par(mfrow=2:1)
w = rnorm(500) # 500 N(0,1) variates
v = filter(w, sides=2, filter=rep(1/3,3)) # moving average
tsplot(w, col=4, main="white noise")
tsplot(v, ylim=c(-3,3), col=4, main="moving average")
```
 
Example 1.9
```r
set.seed(90210)
w = rnorm(250 + 50) # 50 extra to avoid startup problems
x = filter(w, filter=c(1.5,-.75), method="recursive")[-(1:50)]
tsplot(x, main="autoregression", col=4)
```
 
Example 1.10
```r
set.seed(314159265) # so you can reproduce the results
w  = rnorm(200) 
x  = cumsum(w)
wd = w +.3 
xd = cumsum(wd)
tsplot(xd, ylim=c(-2,80), main="random walk", ylab="", col=4)
 clip(0, 200, 0, 80)
 abline(a=0, b=.3, lty=2, col=4) # drift
lines(x, col=6)
 clip(0, 200, 0, 80)
 abline(h=0, col=6, lty=2)
```

Example 1.11
```r
cs = 2*cos(2*pi*(1:500)/50 + .6*pi)
w  = rnorm(500,0,1)
par(mfrow=c(3,1))   
tsplot(cs, ylab="", main = expression(x[t]==2*cos(2*pi*t/50+.6*pi)))
tsplot(cs + w, ylab="", main = expression(x[t]==2*cos(2*pi*t/50+.6*pi)+N(0,1)))
tsplot(cs + 5*w, ylab="", main = expression(x[t]==2*cos(2*pi*t/50+.6*pi)+N(0,25)))
```


[<sub>top</sub>](#table-of-contents)

---



## Chapter 2


Example 2.18
```r
ACF = c(0,0,0,1,2,3,2,1,0,0,0)/3
LAG = -5:5
tsplot(LAG, ACF, type="h", lwd=3, xlab="LAG")   
abline(h=0)
points(LAG[-(4:8)], ACF[-(4:8)], pch=20)
axis(1, at=seq(-5, 5, by=2))         
```

Example 2.25
```r
x = rnorm(100)
y = lag(x,-5) + rnorm(100)
ccf(y, x, ylab="CCovF", type="covariance")
```



Examples 2.27
```r
(r = round( acf1(soi, 6, plot=FALSE), 2)) # sample acf values
# [1] 0.60 0.37 0.21 0.05 -0.11 -0.19
par(mfrow=c(1,2))
tsplot(lag(soi,-1), soi, col=4, type='p')
 legend("topleft", legend=r[1], bg="white", adj=.45, cex = 0.85)
tsplot(lag(soi,-6), soi, col=4, type='p')
 legend("topleft", legend=r[6], bg="white", adj=.25, cex = 0.8)
``` 


Example 2.29
```r
set.seed(101011)
x1 = sample(c(-2,2), 11, replace=TRUE) # simulated coin tosses
x2 = sample(c(-2,2), 101, replace=TRUE)
y1 = 5 + filter(x1, sides=1, filter=c(1,-.5))[-1]
y2 = 5 + filter(x2, sides=1, filter=c(1,-.5))[-1]
tsplot(y1, type="s", col=4, xaxt="n", yaxt="n") # y2 not shown
axis(1, 1:10); axis(2, seq(2,8,2), las=1)
points(y1, pch=21, cex=1.1, bg=6)
acf(y1, lag.max=4, plot=FALSE) 
acf(y2, lag.max=4, plot=FALSE) 
```

Example 2.32
```r
par(mfrow=c(3,1))
acf1(soi, 48, main="Southern Oscillation Index")
acf1(rec, 48, main="Recruitment")
ccf2(soi, rec, 48, main="SOI vs Recruitment")
```

Example 2.33 

```r
set.seed(1492)
num = 120
t = 1:num
X = ts( 2*cos(2*pi*t/12) + rnorm(num), freq=12 )
Y = ts( 2*cos(2*pi*(t+5)/12) + rnorm(num), freq=12 )
Yw = resid(lm(Y~ cos(2*pi*t/12) + sin(2*pi*t/12), na.action=NULL))
par(mfrow=c(3,2))
tsplot(X)
tsplot(Y)
acf1(X, 48)
acf1(Y, 48)
ccf2(X, Y, 24)
ccf2(X, Yw, 24, ylim=c(-.6,.6))
################################################

#  here's another example that's simpler
#  the series are trend stationary with 
#  just a hint of trend - but same result

set.seed(90210)
num = 250  
t   = 1:num
X   = .01*t + rnorm(num,0,2)
Y   = .01*t + rnorm(num)
par(mfrow=c(3,1))
tsplot(cbind(X,Y), spag=TRUE, col=astsa.col(c(4,2),.7), lwd=2, ylab='data')  
ccf2(X, Y,  ylim=c(-.3,.3), col=4, lwd=2)
Yw = resid(lm(Y~t))  # whiten Y by removing trend
ccf2(X, Yw, ylim=c(-.3,.3), col=4, lwd=2)
```





[<sub>top</sub>](#table-of-contents)

---



## Chapter 3



Example 3.1
```r
summary(fit <- lm(salmon~time(salmon), na.action=NULL))
tsplot(salmon, col=4, ylab="USD per KG", main="Salmon Export Price")
abline(fit)
```

Example 3.5
```r
culer = c(rgb(.66,.12,.85), rgb(.12,.66,.85), rgb(.85,.30,.12))
par(mfrow=c(3,1))
tsplot(cmort, main="Cardiovascular Mortality", col=culer[1], type="o", pch=19, ylab="")
tsplot(tempr, main="Temperature", col=culer[2], type="o", pch=19, ylab="")
tsplot(part, main="Particulates", col=culer[3], type="o", pch=19, ylab="")
##
tsplot(cmort, main="", ylab="", ylim=c(20,130), col=culer[1])
lines(tempr, col=culer[2])
lines(part, col=culer[3])   
legend("topright", legend=c("Mortality", "Temperature", "Pollution"), lty=1, lwd=2, col=culer, bg="white")
##
panel.cor <- function(x, y, ...){
 usr <- par("usr"); on.exit(par(usr))
 par(usr = c(0, 1, 0, 1))
 r <- round(cor(x, y), 2)
 text(0.5, 0.5, r, cex = 1.75)   
}
pairs(cbind(Mortality=cmort, Temperature=tempr, Particulates=part), col="dodgerblue3", lower.panel=panel.cor)
##
par(mfrow = 2:1)
plot(tempr, tempr^2) # collinear
cor(tempr, tempr^2)
temp = tempr - mean(tempr)
plot(temp, temp^2) # not collinear
cor(temp, temp^2)
##
temp = tempr - mean(tempr) # center temperature
temp2 = temp^2
trend = time(cmort) # time
fit = lm(cmort~ trend + temp + temp2 + part, na.action=NULL)
summary(fit) # regression results
summary(aov(fit)) # ANOVA table (compare to next line)
summary(aov(lm(cmort~cbind(trend, temp, temp2, part)))) # Table 3.1
num = length(cmort) # sample size
AIC(fit)/num - log(2*pi) # AIC
BIC(fit)/num - log(2*pi) # BIC
```

Example 3.6
```r
fish = ts.intersect( rec, soiL6=lag(soi,-6) )
summary(fit1 <- lm(rec~ soiL6, data=fish, na.action=NULL))
tsplot(resid(fit1), col=4) # residuals
##
library(dynlm)
summary(fit2 <- dynlm(rec~ L(soi,6)))
```

Example 3.10
```r
fit = lm(salmon~time(salmon), na.action=NULL) # the regression
par(mfrow=c(2,1)) # plot transformed data
tsplot(resid(fit), main="detrended salmon price")
tsplot(diff(salmon), main="differenced salmon price")
par(mfrow=c(2,1)) # plot their ACFs
acf1(resid(fit), 48, main="detrended salmon price")
acf1(diff(salmon), 48, main="differenced salmon price")
```


Example 3.11
```r
par(mfrow=c(2,1))
tsplot(diff(gtemp_land), col=4, main="differenced global tmeperature")
mean(diff(gtemp_land)) # drift since 1880
# [1] 0.0143
acf1(diff(gtemp_land))
mean(window(diff(gtemp_land), start=1980)) # drift since 1980
# [1] 0.0329
```

Example 3.12
```r
layout(matrix(1:4,2), widths=c(2.5,1))
par(oma=rep(.2, 4))
tsplot(varve, main="", ylab="", col=4, margin=0)
mtext("varve", side=3, line=.5, cex=1.2, font=2, adj=0)
tsplot(log(varve), main="", ylab="", col=4, margin=0)
mtext("log(varve)", side=3, line=.5, cex=1.2, font=2, adj=0)
qqnorm(varve, main="", col=4); qqline(varve, col=2, lwd=2)
qqnorm(log(varve), main="", col=4); qqline(log(varve), col=2, lwd=2)   
```


Example 3.13
```r
lag1.plot(soi, 12, col=4, cex=1)      # Figure 3.10
lag2.plot(soi, rec, 8, col=4, cex=1)  # Figure 3.11
```

Example 3.14
```r
library(zoo)   # zoo allows easy use of the variable names
dummy = ifelse(soi<0, 0, 1) 
fish = as.zoo(ts.intersect(rec, soiL6=lag(soi,-6), dL6=lag(dummy,-6)))
summary(fit <- lm(rec~ soiL6*dL6, data=fish, na.action=NULL))
plot(fish$soiL6, fish$rec, panel.first=Grid(), col='dodgerblue3')
points(fish$soiL6, fitted(fit), pch=3, col=6) 
lines(lowess(fish$soiL6, fish$rec), col=4, lwd=2)
tsplot(resid(fit))    # not shown
acf1(resid(fit))      # and obviously not noise                       
```


Example 3.15
```r
set.seed(90210) # so you can reproduce these results
x  = 2*cos(2*pi*1:500/50 + .6*pi) + rnorm(500,0,5)
z1 = cos(2*pi*1:500/50)
z2 = sin(2*pi*1:500/50)
summary(fit <- lm(x~ 0 + z1 + z2)) # zero to exclude intercept
par(mfrow=c(2,1))
tsplot(x, col=4)
tsplot(x, ylab=expression(hat(x)), col=astsa.col(4, .5))
lines(fitted(fit), col=2, lwd=2)
```

Example 3.16
```r
w = c(.5, rep(1,11), .5)/12
soif = filter(soi, sides=2, filter=w)
tsplot(soi, col=astsa.col(4,.7), ylim=c(-1, 1.15))
lines(soif, lwd=2, col=4)
# insert
par(fig = c(.65, 1, .75, 1), new = TRUE)
w1 = c(rep(0,20), w, rep(0,20))
plot(w1, type="l", ylim = c(-.02,.1), xaxt="n", yaxt="n", ann=FALSE)
```


Example 3.17
```r
tsplot(soi, col=astsa.col(4,.7), ylim=c(-1, 1.15))
lines(ksmooth(time(soi), soi, "normal", bandwidth=1), lwd=2, col=4)
# insert
par(fig = c(.65, 1, .75, 1), new = TRUE)
gauss <- function(x) { 1/sqrt(2*pi) * exp(-(x^2)/2) }
curve(gauss(x), -3, 3, xaxt="n", yaxt="n", ann=FALSE)

# 
SOI = ts(soi, freq=1)
tsplot(SOI, col=8) # the time scale matters (not shown)
lines(ksmooth(time(SOI), SOI, "normal", bandwidth=12), lwd=2, col=4)
```

Example 3.18
```r
tsplot(soi, col=astsa.col(4,.6))
lines(lowess(soi, f=.05), lwd=2, col=4) # El Ni&ntilde;o cycle
# lines(lowess(soi), lty=2, lwd=2, col=2) # trend (with default span)
##-- trend with CIs using loess --##
lo = predict(loess(soi ~ time(soi)), se=TRUE)
trnd = ts(lo$fit, start=1950, freq=12) # put back ts attributes
lines(trnd, col=6, lwd=2)
L = trnd - qt(0.975, lo$df)*lo$se
U = trnd + qt(0.975, lo$df)*lo$se
xx = c(time(soi), rev(time(soi)))
yy = c(L, rev(U))
polygon(xx, yy, border=8, col=gray(.6, alpha=.4) )
```


Example 3.19
```r
plot(tempr, cmort, xlab="Temperature", ylab="Mortality", col='dodgerblue3', panel.first=Grid())
lines(lowess(tempr,cmort), col=4, lwd=2)
```


Example 3.20
```r
x = window(hor, start=2002)
plot(decompose(x)) # not shown
plot(stl(x, s.window="per")) # not shown
plot(stl(x, s.window=15))  # nicer version below  
##
culer = c(5, 4, 2, 6)
x = window(hor, start=2002)
par(mfrow = c(4,1), cex.main=1)
out = stl(x, s.window=15)$time.series
tsplot(x, main='Hawaiian Occupancy Rate', ylab='% rooms', col=8)
text(x, labels=1:4, col=culer, cex=1.25)
tsplot(out[,1], main="Seasonal", ylab='% rooms',col=8)
text(out[,1], labels=1:4, col=culer, cex=1.25)
tsplot(out[,2], main="Trend", ylab='% rooms', col=8)
text(out[,2], labels=1:4, col=culer, cex=1.25)
tsplot(out[,3], main="Noise", ylab='% rooms', col=8)
text(out[,3], labels=1:4, col=culer, cex=1.25)
```

[<sub>top</sub>](#table-of-contents)

---



## Chapter 4



Example 4.2
```r
par(mfrow=c(2,1))
tsplot(sarima.sim(ar= .9, n=100), ylab="x", col=4, main=expression(AR(1)~~~phi==+.9))
tsplot(sarima.sim(ar=-.9, n=100), ylab="x", col=4, main=expression(AR(1)~~~phi==-.9))
```

Example 4.3
```r
psi = ARMAtoMA(ar = c(1.5, -.75), ma = 0, 50)
par(mfrow=c(2,1))
tsplot(psi, col=4, type='o', pch=19, ylab=expression(psi-weights), xlab='Index', 
        main=expression(AR(2)~~~phi[1]==1.5~~~phi[2]==-.75))
set.seed(8675309)
simulation = sarima.sim(ar=c(1.5,-.75), n=144, S=12)
tsplot(simulation, ylab=expression(X[~t]), col=4, xlab='Year', lwd=2)
```


Examples 4.5
```r
par(mfrow = c(2,1))
tsplot(sarima.sim(ma= .9, n=100), col=4, ylab="x", main=expression(MA(1)~~~theta==+.9))
tsplot(sarima.sim(ma=-.9, n=100), col=4, ylab="x", main=expression(MA(1)~~~theta==-.9))
```


Example 4.10
```r
set.seed(8675309)         # Jenny, I got your number
x = rnorm(150, mean=5)    # generate iid N(5,1)s
arima(x, order=c(1,0,1))  # estimation
```

Example 4.11
```r
AR = c(1, -.3, -.4) # original AR coefs on the left
polyroot(AR)
MA = c(1, .5)       # original MA coefs on the right
polyroot(MA)
```


Example 4.12
```r
round( ARMAtoMA(ar=.8, ma=-.5, 10), 2) # first 10 psi-weights
round( ARMAtoAR(ar=.8, ma=-.5, 10), 2) # first 10 pi-weights
ARMAtoMA(ar=1, ma=0, 20)
```

Example 4.15
```r
ACF = ARMAacf(ar=c(1.5,-.75), ma=0, 50)
tsplot(ACF, type="h", xlab="lag")
abline(h=0, col=8)
```



Example 4.18
```r
ACF  = ARMAacf(ar=c(1.5,-.75), ma=0, 24)[-1]
PACF = ARMAacf(ar=c(1.5,-.75), ma=0, 24, pacf=TRUE)
par(mfrow=1:2)
tsplot(ACF, type="h", xlab="lag", ylim=c(-.8,1))
abline(h=0)
tsplot(PACF, type="h", xlab="lag", ylim=c(-.8,1))
abline(h=0)      
```


Example 4.21
```r
acf2(rec, 48)     # will produce values and a graphic
(regr = ar.ols(rec, order=2, demean=FALSE, intercept=TRUE))
regr$asy.se.coef  # standard errors of the estimates
```


Example 4.24
```r
rec.yw = ar.yw(rec, order=2)
rec.yw$x.mean   # mean estimate
rec.yw$ar       # phi parameter estimates
sqrt(diag(rec.yw$asy.var.coef)) # their standard errors
rec.yw$var.pred # error variance estimate
```


Example 4.25
```r
set.seed(1)
ma1 = sarima.sim(ma = 0.9, n = 50)
acf1(ma1, plot=FALSE)[1]
```

Example 4.27
```r
tsplot(diff(log(varve)), col=4, ylab=expression(nabla~log~X[~t]), main="Transformed Glacial Varves")
acf2(diff(log(varve)))
#
x = diff(log(varve)) # data
r = acf1(x, 1, plot=FALSE) # acf(1)
c(0) -> w -> z -> Sc -> Sz -> Szw -> para # initialize
num = length(x) # = 633
## Estimation
para[1] = (1-sqrt(1-4*(r^2)))/(2*r) # MME
niter = 12
for (j in 1:niter){
for (i in 2:num){ w[i] = x[i]   - para[j]*w[i-1]
                  z[i] = w[i-1] - para[j]*z[i-1]
}
Sc[j]  = sum(w^2)
Sz[j]  = sum(z^2)
Szw[j] = sum(z*w)
para[j+1] = para[j] + Szw[j]/Sz[j]
}
# Results
cbind(iteration=1:niter-1, thetahat=para[1:niter], Sc, Sz)
## Plot conditional SS
c(0) -> w -> cSS
th = -seq(.3, .94, .01)
for (p in 1:length(th)){
for (i in 2:num){ w[i] = x[i] - th[p]*w[i-1]
}
cSS[p] = sum(w^2)
}
tsplot(th, cSS, ylab=expression(S[c](theta)), xlab=expression(theta))
abline(v=para[1:12], lty=2, col=4) # add previous results to plot
points(para[1:12], Sc[1:12], pch=16, col=4)
```




Example 4.28
```r
sarima(diff(log(varve)), p=0, d=0, q=1, no.constant=TRUE)
```

Example 4.31
```r
sarima(rec, p=2, d=0, q=0)  # fit the model
sarima.for(rec, n.ahead=24, p=2, d=0, q=0)
abline(h=61.8585, col=4)    # display estimated mean
```



[<sub>top</sub>](#table-of-contents)

---



## Chapter 5




Example 5.2
```r
sarima(diff(log(varve)), p=0, d=0, q=1, no.constant=TRUE)
# equivalently
sarima(log(varve), p=0, d=1, q=1, no.constant=TRUE)
```

Example 5.3
```r
ARMAtoMA(ar=1, ma=0, 20) # psi-weights for rw
```

Example 5.4
```r
round( ARMAtoMA(ar=c(1.9,-.9), ma=0, 60), 1 ) # ain't no AR(2)
#
set.seed(2001)
x <- sarima.sim(ar=.9, d=1, n=150)
y <- window(x, start=1, end=100)
sarima.for(y, n.ahead = 50, p = 1, d = 1, q = 0, plot.all=TRUE)
text(85, 255, "PAST"); text(115, 255, "FUTURE")
abline(v=100, lty=2, col=4)
lines(x)
```


Example 5.5
```r
set.seed(666)
x = sarima.sim(ma = -0.8, d=1, n = 100)
(x.ima = HoltWinters(x, beta=FALSE, gamma=FALSE)) 
plot(x.ima, main="EWMA")
```


Example 5.6
```r
##-- Figure 5.3 --##
layout(1:2, heights=2:1)
tsplot(gnp, col=4)
acf1(gnp, main="")
##-- Figure 5.4 --##
tsplot(diff(log(gnp)), ylab="GNP Growth Rate", col=4)
abline(h = mean(diff(log(gnp))), col=6)
##-- Figure 5.5 --##
acf2(diff(log(gnp)), main="")
##
sarima(diff(log(gnp)), 0,0,2) # MA(2) on growth rate
sarima(diff(log(gnp)), 1,0,0) # AR(1) on growth rate
#
round( ARMAtoMA(ar=.35, ma=0, 10), 3) # print psi-weights
```


Example 5.7 
```r
sarima(diff(log(gnp)), 0, 0, 2) # MA(2) fit with diagnostics
```


Example 5.8
```r
sarima(log(varve), 0, 1, 1, no.constant=TRUE) # ARIMA(0,1,1)
sarima(log(varve), 1, 1, 1, no.constant=TRUE) # ARIMA(1,1,1)
```


Example 5.9 
```r
uspop = c(75.995, 91.972, 105.711, 123.203, 131.669,150.697, 179.323, 203.212, 226.505, 249.633, 281.422, 308.745)
uspop = ts(uspop, start=1900, freq=.1)
t = time(uspop) - 1955
reg = lm( uspop~ t+I(t^2)+I(t^3)+I(t^4)+I(t^5)+I(t^6)+I(t^7)+I(t^8) )
b = as.vector(reg$coef)
g = function(t){  b[1] + b[2]*(t-1955) + b[3]*(t-1955)^2 + b[4]*(t-1955)^3 + b[5]*(t-1955)^4 + b[6]*(t-1955)^5 + b[7]*(t-1955)^6 + b[8]*(t-1955)^7 + b[9]*(t-1955)^8
}
par(mar=c(2,2.5,.5,0)+.5, mgp=c(1.6,.6,0))
curve(g, 1900, 2024, ylab="Population", xlab="Year", main="U.S. Population by Official Census", panel.first=Grid(), cex.main=1, font.main=1, col=4)
abline(v=seq(1910,2020,by=20), lty=1, col=gray(.9))
points(time(uspop), uspop, pch=21, bg=rainbow(12), cex=1.25)
mtext(expression(""%*% 10^6), side=2, line=1.5, adj=.95)
axis(1, seq(1910,2020,by=20), labels=TRUE)
```



Example 5.10 
```r
sarima(diff(log(gnp)), 1, 0, 0) # AR(1)
sarima(diff(log(gnp)), 0, 0, 2) # MA(2)
```




Example 5.11 
```r
set.seed(111111)
SAR = sarima.sim(sar=.9, S=12, n=37) + 50
layout(matrix(c(1,2, 1,3), nc=2), heights=c(1.5,1))
tsplot(SAR, type="c", xlab="Year")
 abline(v=1:3, col=4, lty=2)
 Months = c("J","F","M","A","M","J","J","A","S","O","N","D")
 points(SAR, pch=Months, cex=1.35, font=4, col=1:6)

phi  = c(rep(0,11),.9)
ACF  = ARMAacf(ar=phi, ma=0, 100)[-1] # [-1] removes 0 lag
PACF = ARMAacf(ar=phi, ma=0, 100, pacf=TRUE)
 LAG = 1:100/12
tsplot(LAG, ACF, type="h", xlab="LAG", ylim=c(-.04,1))
 abline(h=0, col=8)
tsplot(LAG, PACF, type="h", xlab="LAG", ylim=c(-.04,1))
 abline(h=0, col=8)
```


Example 5.12 
```r
##-- Figure 5.10 --##
phi = c(rep(0,11),.8)
ACF = ARMAacf(ar=phi, ma=-.5, 50)[-1]
PACF = ARMAacf(ar=phi, ma=-.5, 50, pacf=TRUE)
LAG = 1:50/12
par(mfrow=c(1,2))
tsplot(LAG,  ACF, type='h', xlab='LAG')
tsplot(LAG, PACF, type='h', xlab='LAG')
##-- birth series --##
tsplot(birth) # monthly number of births in US
acf2( diff(birth) ) # P/ACF of the differenced birth rate

##-- seasonal persistence --##
x = window(hor, start=2002)
par(mfrow = c(2,1))
tsplot(x, main="Hawaiian Quarterly Occupancy Rate", ylab=" % rooms", ylim=c(62,86), col=gray(.7))
text(x, labels=1:4, col=c(3,4,2,6), cex=.8)
Qx = stl(x,15)$time.series[,1]
tsplot(Qx, main="Seasonal Component", ylab=" % rooms", ylim=c(-4.7,4.7), col=gray(.7))
text(Qx, labels=1:4, col=c(3,4,2,6), cex=.8)
```


Example 5.15 
```r
par(mfrow=c(2,1))
tsplot(cardox, col=4, ylab=expression(CO[2]))
title("Monthly Carbon Dioxide Readings - Mauna Loa Observatory ", cex.main=1)
tsplot(diff(diff(cardox,12)), col=4,
ylab=expression(nabla~nabla[12]~CO[2]))

acf2(diff(diff(cardox,12)))

sarima(cardox, p=0,d=1,q=1, P=0,D=1,Q=1,S=12)
sarima(cardox, p=1,d=1,q=1, P=0,D=1,Q=1,S=12)

sarima.for(cardox, 60, 1,1,1, 0,1,1,12)
abline(v=2018.9, lty=6)
##-- for comparison --##
sarima.for(cardox, 60, 0,1,1, 0,1,1,12) # not shown
```


Example 5.16 
```r
trend = time(cmort); temp = tempr - mean(tempr); temp2 = temp^2
fit = lm(cmort~trend + temp + temp2 + part, na.action=NULL)
acf2(resid(fit), 52) # implies AR2
sarima(cmort, 2,0,0, xreg=cbind(trend, temp, temp2, part) )
```


Example 5.17 
```r
library(zoo)
lag2.plot(Hare, Lynx, 5) # lead-lag relationship
pp = as.zoo(ts.intersect(Lynx, HareL1 = lag(Hare,-1)))
summary(reg <- lm(pp$Lynx~ pp$HareL1)) # results not displayed
acf2(resid(reg)) # in Figure 5.11
( reg2 = sarima(pp$Lynx, 2,0,0, xreg=pp$HareL1 ))
prd = Lynx - resid(reg2$fit) # prediction (resid = obs - pred)
prde = sqrt(reg2$fit$sigma2) # prediction error
tsplot(prd, lwd=2, col=rgb(0,0,.9,.5), ylim=c(-20,90), ylab="Lynx")
points(Lynx, pch=16, col=rgb(.8,.3,0))
x = time(Lynx)[-1]
xx = c(x, rev(x))
yy = c(prd - 2*prde, rev(prd + 2*prde))
polygon(xx, yy, border=8, col=rgb(.4, .5, .6, .15))
mtext(expression(""%*% 10^3), side=2, line=1.5, adj=.975)
legend("topright", legend=c("Predicted", "Observed"), lty=c(1,NA), lwd=2, pch=c(NA,16), col=c(4,rgb(.8,.3,0)), cex=.9)
```




[<sub>top</sub>](#table-of-contents)

---



## Chapter 6





Aliasing

```r
t = seq(0, 24, by=.01)
X = cos(2*pi*t*1/2) # 1 cycle every 2 hours
tsplot(t, X, xlab="Hours")
T = seq(1, length(t), by=250) # observed every 2.5 hrs
points(t[T], X[T], pch=19, col=4)
lines(t, cos(2*pi*t/10), col=4)
axis(1, at=t[T], labels=FALSE, lwd.ticks=2, col.ticks=2)
abline(v=t[T], col=rgb(1,0,0,.2), lty=2)
```

Example 6.1
```r
x1 = 2*cos(2*pi*1:100*6/100) + 3*sin(2*pi*1:100*6/100)
x2 = 4*cos(2*pi*1:100*10/100) + 5*sin(2*pi*1:100*10/100)
x3 = 6*cos(2*pi*1:100*40/100) + 7*sin(2*pi*1:100*40/100)
x = x1 + x2 + x3
par(mfrow=c(2,2))
tsplot(x1, ylim=c(-10,10), main=expression(omega==6/100~~~A^2==13))
tsplot(x2, ylim=c(-10,10), main=expression(omega==10/100~~~A^2==41))
tsplot(x3, ylim=c(-10,10), main=expression(omega==40/100~~~A^2==85))
tsplot(x, ylim=c(-16,16), main="sum")

# periogoram -- after example
P = Mod(fft(x)/sqrt(100))^2 # periodogram
sP = (4/100)*P   # scaled peridogram
Fr = 0:99/100    # fundamental frequencies
tsplot(Fr, sP, type="o", xlab="frequency", ylab="scaled periodogram", col=4, ylim=c(0,90))
abline(v=.5, lty=5)
abline(v=c(.1,.3,.7,.9), lty=1, col=gray(.9))
axis(side=1, at=seq(.1,.9,by=.2))
```

Example 6.5
```r
par(mfrow=c(3,2))
for(i in 4:9){
mvspec(fmri1[,i], main=colnames(fmri1)[i], ylim=c(0,3), xlim=c(0,.2), col=5, lwd=2, type="o", pch=20)
abline(v=1/32, col=4, lty=5) # stimulus frequency
}
```


Example 6.7, 6.9, 6.10
```r
par(mfrow=c(3,1))
arma.spec(main="White Noise", col=4)
arma.spec(ma=.5, main="Moving Average", col=4)
arma.spec(ar=c(1,-.9), main="Autoregression", col=4)
```

Example 6.12
```r
par(mfrow=c(3,1))
tsplot(soi, col=4, main="SOI")
tsplot(diff(soi), col=4, main="First Difference")
k = kernel("modified.daniell", 6) # MA weights
tsplot(kernapply(soi, k), col=4, main="Seasonal Moving Average")
##-- frequency responses --##
par(mfrow=c(2,1))
w = seq(0, .5, by=.01)
FRdiff = abs(1-exp(2i*pi*w))^2
tsplot(w, FRdiff, xlab="frequency", main="High Pass Filter")
u = cos(2*pi*w)+cos(4*pi*w)+cos(6*pi*w)+cos(8*pi*w)+cos(10*pi*w)
FRma = ((1 + cos(12*pi*w) + 2*u)/12)^2
tsplot(w, FRma, xlab="frequency", main="Low Pass Filter")
```





[<sub>top</sub>](#table-of-contents)

---



## Chapter 7



DFTs
```r
(dft = fft(1:4)/sqrt(4))
(idft = fft(dft, inverse=TRUE)/sqrt(4))
```


Example 7.4
```r
par(mfrow=c(2,1)) # raw periodogram
mvspec(soi, col=rgb(.05,.6,.75), lwd=2)
rect(1/7, -1e5, 1/3, 1e5, density=NA, col=gray(.5,.2))
abline(v=1/4, lty=2, col="dodgerblue")
mtext("1/4", side=1, line=0, at=.25, cex=.75)
mvspec(rec, col=rgb(.05,.6,.75), lwd=2)
rect(1/7, -1e5, 1/3, 1e5, density=NA, col=gray(.5,.2))
abline(v=1/4, lty=2, col="dodgerblue")
mtext("1/4", side=1, line=0, at=.25, cex=.75)  

#  log redux
par(mfrow=c(2,1)) # raw periodogram
mvspec(soi, col=rgb(.05,.6,.75), lwd=2, log='y')
rect(1/7, 1e-5, 1/3, 1e5, density=NA, col=gray(.5,.2))
abline(v=1/4, lty=2, col="dodgerblue")
mtext("1/4", side=1, line=0, at=.25, cex=.75)
mvspec(rec, col=rgb(.05,.6,.75), lwd=2, log='y')
rect(1/7, 1e-5, 1/3, 1e5, density=NA, col=gray(.5,.2))
abline(v=1/4, lty=2, col="dodgerblue")
mtext("1/4", side=1, line=0, at=.25, cex=.75) 
```


Periodogram... Bad! 
```r
u = mvspec(rnorm(1000), col=5) # periodogram
abline(h=1, col=2, lwd=5)      # true spectrum
lines(u$freq, filter(u$spec, filter=rep(1,101)/101, circular=TRUE), col=4, lwd=2) # add the smooth
```

Example 7.5
```r
par(mfrow=c(2,1))
soi.ave = mvspec(soi, spans=9, col=5, lwd=2)
rect(1/7, -1e5, 1/3, 1e5, density=NA, col=gray(.5,.2))
abline(v=.25, lty=2, col=4)
mtext("1/4", side=1, line=0, at=.25, cex=.75)
rec.ave = mvspec(rec, spans=9, col=5, lwd=2)
rect(1/7, -1e5, 1/3, 1e5, density=NA, col=gray(.5,.2))
abline(v=.25, lty=2, col=4)
mtext("1/4", side=1, line=0, at=.25, cex=.75)

##-- redo on log scale with CIs --##
par(mfrow=c(2,1))
soi.ave = mvspec(soi, spans=9, col=5, lwd=2, log='yes')  
rect(1/7, 1e-5, 1/3, 1e5, density=NA, col=gray(.5,.2))                 
abline(v=.25, lty=2, col="dodgerblue")
mtext("1/4", side=1, line=0, at=.25, cex=.75)
rec.ave = mvspec(rec, spans=9, col=5, lwd=2, log='yes')  
rect(1/7, 1e-5, 1/3, 1e5, density=NA, col=gray(.5,.2))                 
abline(v=.25, lty=2, col="dodgerblue")
mtext("1/4", side=1, line=0, at=.25, cex=.75)
```

Example 7.6
```r
y = ts(rev(1:100 %% 20), freq=20) # sawtooth signal
par(mfrow=2:1)
tsplot(1:100, y, ylab="sawtooth signal", col=4)
mvspec(y, main="", ylab="periodogram", col=5, xlim=c(0,7))  
```

Example 7.7
```r
(dm = kernel("modified.daniell", c(3,3))) # for a list
# the figure with both kernels
par(mfrow=1:2, mar=c(3,3,2,1), mgp=c(1.6,.6,0))
plot(kernel("modified.daniell", c(3,3)), ylab=expression(h[~k]), cex.main=1, col=4,  panel.first=Grid())
plot(kernel("modified.daniell", c(3,3,3)), ylab=expression(h[~k]), cex.main=1, col=4,  panel.first=Grid())
#
par(mfrow=c(2,1))
sois = mvspec(soi, spans=c(7,7), taper=.1, col=5, lwd=2)
rect(1/7, -1e5, 1/3, 1e5, density=NA, col=gray(.5,.2))
abline(v=.25, lty=2, col=4)
mtext("1/4", side=1, line=0, at=.25, cex=.75)
recs = mvspec(rec, spans=c(7,7), taper=.1, col=5, lwd=2)
rect(1/7, -1e5, 1/3, 1e5, density=NA, col=gray(.5,.2))
abline(v=.25, lty=2, col=4)
mtext("1/4", side=1, line=0, at=.25, cex=.75)
sois$Lh
sois$bandwidth
# to find the peaks
sois$details[1:45,]

##-- for the logs - not shown in the text --##
par(mfrow=c(2,1))
sois = mvspec(soi, spans=c(7,7), taper=.1, col=5, lwd=2, log='yes')
rect(1/7, 1e-5, 1/3, 1e5, density=NA, col=gray(.5,.2))
abline(v=.25, lty=2, col=4)
mtext("1/4", side=1, line=0, at=.25, cex=.75)
recs = mvspec(rec, spans=c(7,7), taper=.1, col=5, lwd=2, log='yes')
rect(1/7, 1e-5, 1/3, 1e5, density=NA, col=gray(.5,.2))
abline(v=.25, lty=2, col=4)
mtext("1/4", side=1, line=0, at=.25, cex=.75)
```


Tapering
```r
w = seq(-.04,.04,.0001); n=480; u=0
for (i in -4:4){ 
 k = i/n
 u = u + sin(n*pi*(w+k))^2 / sin(pi*(w+k))^2
}
fk = u/(9*480)
u=0; wp = w+1/n; wm = w-1/n
for (i in -4:4){
 k  = i/n; wk = w+k; wpk = wp+k; wmk = wm+k
 z  = complex(real=0, imag=2*pi*wk)
 zp = complex(real=0, imag=2*pi*wpk)
 zm = complex(real=0, imag=2*pi*wmk)
 d  = exp(z)*(1-exp(z*n))/(1-exp(z))
 dp = exp(zp)*(1-exp(zp*n))/(1-exp(zp))
 dm = exp(zm)*(1-exp(zm*n))/(1-exp(zm))
 D  = .5*d - .25*dm*exp(pi*w/n)-.25*dp*exp(-pi*w/n)
 D2 = abs(D)^2
 u  = u + D2
}
sfk = u/(480*9)
par(mfrow=c(1,2))
plot(w, fk, type="l", ylab="", xlab="frequency", main="Without Tapering", yaxt="n")
mtext(expression("|"), side=1, line=-.20, at=c(-0.009375, .009375), cex=1.5, col=2)
segments(-4.5/480, -2, 4.5/480, -2 , lty=1, lwd=3, col=2)
plot(w, sfk, type="l", ylab="",xlab="frequency", main="With Tapering", yaxt="n")
mtext(expression("|"), side=1, line=-.20, at=c(-0.009375, .009375), cex=1.5, col=2)
segments(-4.5/480, -.78, 4.5/480, -.78, lty=1, lwd=3, col=2) 
```



Example 7.8
```r
s0 = mvspec(soi, spans=c(7,7), plot=FALSE)             # no taper
s50 = mvspec(soi, spans=c(7,7), taper=.5, plot=FALSE)  # full taper
tsplot(s50$freq, s50$spec, log="y", type="l", ylab="spectrum", xlab="frequency") 
lines(s0$freq, s0$spec, lty=2) 
abline(v=.25, lty=2, col=8)
mtext('1/4',side=1, line=0, at=.25, cex=.9)
legend(5,.04, legend=c('full taper', 'no taper'), lty=1:2)

text(1.42, 0.04, 'leakage', cex=.8)
arrows(1.4, .035, .75, .009, length=0.05,angle=30)   
arrows(1.4, .035, 1.21, .0075, length=0.05,angle=30)
par(fig = c(.65, 1, .65, 1),  new = TRUE, cex=.5,  mgp=c(0,-.1,0), tcl=-.2)
taper <- function(x) { .5*(1+cos(2*pi*x))  }
 x <- seq(from = -.5, to = .5, by = 0.001)
plot(x, taper(x), type = "l",  lty = 1,  yaxt='n', ann=FALSE)
```


Example 7.10
```r
# AR spectrum - AIC picks order=15
u <- spec.ic(soi,  detrend=TRUE, col=4, lwd=2, nxm=4)  
# plot AIC and BIC
dev.new()
tsplot(0:30, u[[1]][,2:3], type='o', col=2:3, xlab='ORDER', nxm=5, lwd=2, gg=TRUE)  
```


Example 7.12
```r
sr = mvspec(cbind(soi,rec), kernel("daniell",9), plot.type="coh")
sr$df                     # df = 35.8625
f = qf(.999, 2, sr$df-2)  # f = 8.529792
C = f/(18+f)              # C = 0.3188779
abline(h = C)
```



[<sub>top</sub>](#table-of-contents)

---



## Chapter 8



Example 8.1
```r
res = resid( sarima(diff(log(gnp)), 1,0,0, details=FALSE)$fit )
acf2(res^2, 20)
#
library(fGarch)
gnpr = diff(log(gnp))
summary( garchFit(~arma(1,0) + garch(1,0), data = gnpr) )
```


Example 8.2
```r
library(xts)
djiar = diff(log(djia$Close))[-1]
acf2(djiar) # exhibits some autocorrelation
u = resid( sarima(djiar, 1,0,0, details=FALSE)$fit )
acf2(u^2)   # oozes autocorrelation
library(fGarch)
summary(djia.g <- garchFit(~arma(1,0)+garch(1,1), data=djiar, cond.dist="std"))
plot(djia.g, which=3)   
```


Example 8.3
```r
lapply( c("xts", "fGarch"), library, char=TRUE) # load 2 packages in one line - amazing!
djiar = diff(log(djia$Close))[-1]
summary(djia.ap <- garchFit(~arma(1,0)+aparch(1,1), data=djiar, cond.dist="std"))
plot(djia.ap)   # to see all plot options 
```

Example 8.4
```r
layout(1:2)
acf1(cumsum(rnorm(634)), 100, main="Series: random walk")
acf1(log(varve), 100, ylim=c(-.1,1)) 
#
library(tseries)
adf.test(log(varve), k=0) # DF test
adf.test(log(varve))      # ADF test
pp.test(log(varve))       # PP test
```


Example 8.5
```r
d = 0.3727893; p = c(1)
for (k in 1:30){ p[k+1] = (k-d)*p[k]/(k+1) }
tsplot(1:30, p[-1], ylab=expression(pi(d)), lwd=2, xlab="Index", type="h", col="dodgerblue3")
library(arfima)
summary(varve.fd <- arfima(log(varve), order = c(0,0,0)))
# residuals
innov = resid(varve.fd)
tsplot(innov[[1]]) # not shown
par(mfrow=2:1)
acf1(resid(sarima(log(varve),1,1,1, details=FALSE)$fit), main="ARIMA(1,1,1)")
acf1(innov[[1]], main="Frac Diff")
```

Example 8.8
```r
u = ssm(gtemp_land, A=1, alpha=.01, phi=1, sigw=.01, sigv=.1)
tsplot(gtemp_land, col="dodgerblue3", type="o", pch=20, ylab="Temperature Deviations")
lines(u$Xs, col=6, lwd=2)
xx = c(time(u$Xs), rev(time(u$Xs)))
yy = c(u$Xs-2*sqrt(u$Ps), rev(u$Xs+2*sqrt(u$Ps)))
polygon(xx, yy, border=8, col=gray(.6, alpha=.25) )
```


Example 8.9
```r
ccf2(cmort, part) # Figure 8.7
acf2(diff(cmort)) # Figure 8.8 implies AR(1)
u = sarima(cmort, 1, 1, 0, no.constant=TRUE) # fits well
cmortw = resid(u$fit)  
phi = as.vector(u$fit$coef) # -.5064
# filter particluates the same way
partf = filter(diff(part), filter=c(1, -phi), sides=1)
## -- now line up the series - this step is important --##
both = ts.intersect(cmortw, partf) # line them up
Mw = both[,1] # cmort whitened
Pf = both[,2] # part filtered
ccf2(Mw, Pf) # Figure 8.9 
```



-- Section 8.6 --
```r
# data
set.seed(101010)
e = rexp(150, rate=.5); u = runif(150,-1,1); de = e*sign(u)
dex = 50 + arima.sim(n=100, list(ar=.95), innov=de, n.start=50)
layout(matrix(1:2, nrow=1), widths=c(5,2))
tsplot(dex, col=4, ylab=expression(X[~t]))
# density - standard Laplace vs normal
f = function(x) { .5*dexp(abs(x), rate = 1/sqrt(2))}
curve(f, -5, 5, panel.first=Grid(), col=4, ylab="f(w)", xlab="w")
par(new=TRUE)
curve(dnorm, -5, 5, ylab="", xlab="", yaxt="no", xaxt="no", col=2) 
#
fit = ar.yw(dex, order=1)
round(cbind(fit$x.mean, fit$ar, fit$var.pred), 2)
#
set.seed(111)
phi.yw = c()
for (i in 1:1000){
 e  = rexp(150, rate=.5)
 u  = runif(150,-1,1)
 de = e*sign(u)
 x  = 50 + arima.sim(n=100, list(ar=.95), innov=de, n.start=50)
 phi.yw[i] = ar.yw(x, order=1)$ar
} 
#
set.seed(666) # not that 666
fit = ar.yw(dex, order=1) # assumes the data were retained
m = fit$x.mean # estimate of mean
phi = fit$ar # estimate of phi
nboot = 500 # number of bootstrap replicates
resids = fit$resid[-1] # the 99 residuals
x.star = dex # initialize x*
phi.star.yw = c()
# Bootstrap
for (i in 1:nboot) {
 resid.star = sample(resids, replace=TRUE)
 for (t in 1:99)
 {
  x.star[t+1] = m + phi*(x.star[t]-m) + resid.star[t]
 }
 phi.star.yw[i] = ar.yw(x.star, order=1)$ar
}
# Picture
culer = rgb(0,.5,.5,.5)
hist(phi.star.yw, 15, main="", prob=TRUE, xlim=c(.65,1.05),
ylim=c(0,14), col=culer, xlab=expression(hat(phi)))
lines(density(phi.yw, bw=.02), lwd=2) # from previous simulation
u = seq(.75, 1.1, by=.001) # normal approximation
lines(u, dnorm(u, mean=.96, sd=.03), lty=2, lwd=2)
legend(.65, 14, legend=c("true distribution", "bootstrap distribution", "normal approximation"), bty="n", lty=c(1,0,2), lwd=c(2,0,2), col=1, pch=c(NA,22,NA), pt.bg=c(NA,culer,NA), pt.cex=2.5)
# CIs 
alf = .025 # 95% CI
quantile(phi.star.yw, probs = c(alf, 1-alf))
quantile(phi.yw, probs = c(alf, 1-alf))
n=100; phi = fit$ar; se = sqrt((1-phi)/n)
c( phi - qnorm(1-alf)*se, phi + qnorm(1-alf)*se )
```

Example 8.10
```r
tsplot(flu, type="c", ylab="Influenza Deaths per 10,000")
Months = c("J","F","M","A","M","J","J","A","S","O","N","D")
points(flu, pch=Months, cex=.8, font=4, col=c(4,2,3,6))

# Start analysis
dev.new()
dflu = diff(flu)
lag1.plot(dflu, corr=FALSE) # scatterplot with lowess fit
thrsh = .05 # threshold
Z = ts.intersect(dflu, lag(dflu,-1), lag(dflu,-2), lag(dflu,-3), lag(dflu,-4) )
ind1 = ifelse(Z[,2] < thrsh, 1, NA) # indicator < thrsh
ind2 = ifelse(Z[,2] < thrsh, NA, 1) # indicator >= thrsh
X1 = Z[,1]*ind1
X2 = Z[,1]*ind2
summary(fit1 <- lm(X1~ Z[,2:5]) ) # case 1
summary(fit2 <- lm(X2~ Z[,2:5]) ) # case 2
D = cbind(rep(1, nrow(Z)), Z[,2:5]) # design matrix
p1 = D %*% coef(fit1) # get predictions
p2 = D %*% coef(fit2)
prd = ifelse(Z[,2] < thrsh, p1, p2)

# Figure 8.11
dev.new()
tsplot(prd, ylim=c(-.5,.5), ylab=expression(nabla~flu[~t]), lwd=2, col=4)
prde1 = sqrt(sum(resid(fit1)^2)/df.residual(fit1))
prde2 = sqrt(sum(resid(fit2)^2)/df.residual(fit2))
prde = ifelse(Z[,2] < thrsh, prde1, prde2)
x = time(dflu)[-(1:4)]
 xx = c(x, rev(x))
 yy = c(prd - 2*prde, rev(prd + 2*prde))
polygon(xx, yy, border=8, col=gray(.8, .3))
abline(h=.05, col=4, lty=6)
points(dflu, pch=16, col=2)
#
dev.new()
par(mar=c(2.5,2.5,0,0)+.5, mgp=c(1.6,.6,0))
U = matrix(Z, ncol=5) # Z was created in the analysis above
culer = c(rgb(0,1,0,.4), rgb(0,0,1,.4))
culers = ifelse(U[,2]<.05, culer[1], culer[2])
plot(U[,2], U[,1], panel.first=Grid(), pch=21, cex=1.1, bg=culers, xlab=expression(nabla~flu[~t-1]), ylab=expression(nabla~flu[~t]))
lines(lowess(U[,2], U[,1], f=2/3), col=6)
abline(v=.05, lty=2, col=4)

##- alternate method
library(tsDyn) # load package - install it if you don"t have it
# vignette("tsDyn") # for package details
(u = setar(dflu, m=4, thDelay=0, th=.05)) # fit model and view results
(u = setar(dflu, m=4, thDelay=0)) # let program fit threshold (=.036)
AIC(u) # if you want to try other models; m=3 works well too
plot(u) # graphics - ?plot.setar for information
```

[<sub>top</sub>](#table-of-contents)

---
---
