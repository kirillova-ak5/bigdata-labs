
##########################################
Prony <- function(x,m,dt=1) {
  n = length(x)
  
  X<-matrix(nrow = n-m, ncol=m)
  for (i in 0:(m-1)) {
    X[,m-i] <- x[(1+i):(n-m+i)]
  }
  
  a = lm(x[(m+1):n] ~ 0 + X)$coefficients
  a<-append(-rev(a),1)
  
  zk <- polyroot(a)
  Z<-matrix(nrow = n, ncol=m)
  for (i in 1:n) {
    Z[i,] <- zk^i
  }
  #couldve used zk^[1:n] but was too afraid
  
  #I HATE R
  Zmod <- rbind(cbind(Re(Z), -Im(Z)),cbind(Im(Z), Re(Z)))
  xmod <- append(Re(x), Im(x))
  hmod <- lm(xmod ~ 0 + Zmod)$coefficients
  h <- hmod[1:m]+1i*hmod[(m+1):(2*m)]
  #END OF COSTYL(C)
  
  
   

  
  lam <- log(abs(zk))/dt
  ome <- 1/2/pi/dt*sapply(zk, function(z) atan(Im(z)/Re(z)))
  A <- abs(h)
  phi <- sapply(h, function(z) atan(Im(z)/Re(z)))

  return (Z%*%h)
}
########################################

Herst <- function(x) {
  xmean = mean(x)
  N=min(100,length(x)%/%4)
  R=N:length(x)
  s=N:length(x)
  dev_=1:length(x)
  for (i in 1:length(x)){
    dev_[i] = sum(x[1:i])-i*xmean
  }
  
  
  for (n in R) {
    s[n-N+1] = sd(x[1:n])*sqrt(n/(n-1))
    dev_max = max(dev_[1:n])
    dev_min = min(dev_[1:n])
    R[n-N+1]=dev_max-dev_min
  }
  # print("x=") 
  # print(x)
  # print("dev=") 
  # print(dev_)
  # print("s=") 
  # print(s)
  # print("R=") 
  # print(R)
  return (R/s)
}

##################################################


data=read.table('wr96018a1.txt', header = FALSE)
a=Herst(data[,5])

i=log(100:2192)
plot(1:2192,data[,5], type="l")
plot(100:2192, a, type="l")
plot(i, log(a), type="l")
h=lm(log(a) ~ i)$coefficients
r=lm(log(a) ~ i)$residuals
lines(i, i*h[2]+h[1], type="l", lty="dashed")
mtext(sprintf("h = %f, s = %g",h[2],sd(r)))


library(zoo)
par=11
b=data[,5]
b[1:par]=mean(b[1:par])
b[(2192-par+1):2192]=mean(b[(2192-par+1):2192])
b[(par+1):(2192-par)]=rollmean(data[,5],2*par+1)
plot(data[,5], pch=".")
lines(1:2192,b)
f=fft(b)
plot(1:length(f)-1, abs(f/length(f)), type="l")
plot(0:50, abs(f[1:51]/length(f)), type="l")
abline(v = 1/365*length(f), lty='dashed')
freq=round(1/365*length(f))

meanFourier = abs(f[1])/2192
Fourier=Re(f[freq+1])/2192*cos(freq*(1-1:2192)/2192*2*pi) +
  Im(f[freq+1])/2192*sin(freq*(1-1:2192)/2192*2*pi)
afterFourier = b - meanFourier - Fourier

plot(type="l",b)
points(1:2192, data[,5], type="p", pch=46)
lines(1:2192,Fourier, col ='blue')
lines(afterFourier-Fourier, col = 'red') # получено эмпирически 
# Почему надо вычитать два Фурье? Потому что частота 2pi-k перехватила половину амплитуды?






h=0.02
i = 1:200
c_args = rep.int(0,length(i))
#pk = 4*pi*h*i+p/k
for (k in 1:3) {
  c_args <- c_args+k*exp(-h*i/k)*cos(c_args+4*pi*k*h*i+pi/k)
}

plot.new()

#plot.window(xlim=c(0,10), ylim=c(-1,1))
axis(2, pos = 0)
axis(1, pos = 0)
res = Re(Prony(c_args, 30))
plot(i, c_args)
plot.xy(xy.coords(i,res), type="l",pch=NA_integer_,lty="solid",lwd=1.5,col=rgb(0.1,0.6,0,1,0))
#plot.xy(xy.coords(i,res), type="l",pch=NA_integer_,lty="dashed",lwd=1.5,col=rgb(0.1,0.6,0,1,0))

#plot(i, Proni(c_args, 3))
#















# unused
# 
# pt1 = 220
# pt2 = 280
# pt3 = 820
# 
# ind1 = log(100:pt1)
# ind2 = log(pt1:pt2)
# ind3 = log(pt2:pt3)
# ind4 = log(pt3:2192)
# 
# a1 = log(Herst((data[,5])[1:(pt1)]))
# ind1 = log(55:pt1)
# a2 = log(a[(pt1-99):(pt2-99)])
# a3 = log(a[(pt2-99):(pt3-99)])
# a4 = log(a[(pt3-99):length(a)])
# 
# r=1:4
# h1=lm(a1 ~ ind1)$coefficients
# h2=lm(a2 ~ ind2)$coefficients
# h3=lm(a3 ~ ind3)$coefficients
# h4=lm(a4 ~ ind4)$coefficients
# r[1]=sd(lm(a1 ~ ind1)$residuals)
# r[2]=sd(lm(a2 ~ ind2)$tol)
# r[3]=sd(lm(a3 ~ ind3)$tol)
# r[4]=sd(lm(a4 ~ ind4)$tol)
# 
# plot(ind1, a1,  type="l",pch=NA_integer_,lty="solid")
# lines(ind1, ind1*h1[2]+h1[1],  type="l",pch=NA_integer_,lty="dashed")
# mtext(sprintf("h = %f, s = %g",h1[2],r[1]))
# 
# plot(ind2, a2,  type="l",pch=NA_integer_,lty="solid")
# lines(ind2, ind2*h2[2]+h2[1],  type="l",pch=NA_integer_,lty="dashed")
# mtext(sprintf("h = %f, s = %g",h2[2],r[2]))
# 
# plot(ind3, a3,  type="l",pch=NA_integer_,lty="solid")
# lines(ind3, ind3*h3[2]+h3[1],  type="l",pch=NA_integer_,lty="dashed")
# mtext(sprintf("h = %f, s = %g",h3[2],r[3]))
# 
# plot(ind4, a4,  type="l",pch=NA_integer_,lty="solid")
# lines(ind4, ind4*h4[2]+h4[1],  type="l",pch=NA_integer_,lty="dashed")
# mtext(sprintf("h = %f, s = %g",h4[2],r[4]))










