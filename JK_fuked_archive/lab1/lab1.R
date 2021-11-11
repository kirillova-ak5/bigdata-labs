#1
x=seq(0, 3*pi, by=0.01)
y1=sin(x)
y2=cos(x)

vals = (0:6)
names = strsplit(toString(vals), ", ")[[1]]
names
res <- paste(names, "/2 pi", sep='')
plot.new()
plot.window(xlim=c(0,10), ylim=c(-1,1))
axis(2, pos = 0)
axis(1, pos = 0, at = vals*pi/2, labels=res)
plot.xy(xy.coords(x,y1), type="l", pch=NA_integer_,lty="solid",lwd=1.5,col=rgb(0.6,0.1,0,1,0))
plot.xy(xy.coords(x,y2), type="l",pch=NA_integer_,lty="solid",lwd=1.5,col=rgb(0.1,0.6,0,1,0))

#2
a=-10:5
b=-5:10
#3
f=0:15
f[seq(2,16,by=2)]=b[seq(2,16,by=2)]
f[seq(1,15,by=2)]=a[seq(1,15,by=2)]
e=sort(f)
#4
norm_v <- function(x) sqrt(sum(x^2))
norm_v(a)
norm_v(b)
norm_v(e)
#5
A=matrix(1:100, nrow=10, ncol = 10)
rs = apply(A[1:10,], 1, sum)
rc = apply(A[,1:10], 2, sum)
A*rs
A*rc
B=A[1:5,1:5]
#6
my_factorial <- function(x) prod(1:x)
my_factorial(3)
my_factorial(4)
my_factorial(10)
#7
vec = rep.int(0,5)
for (i in 1:length(vec)) {
  vec[i] <- as.numeric(readline())
}
1
2
3
4
5
m1 = min(vec)
m5 = max(vec)
s = sum(vec)