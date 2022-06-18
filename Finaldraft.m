% This code is for population forecasting problem using multivariuate
% regression
clear
I_val = []; Final_prediction_val=[]; M_val = [];
a={'Benin', 'Burkina Faso', 'Ghana', 'Guinea', 'Ivory Coast', 'Mali', 'Niger', 'Nigeria', 'Senegal'};
k=1:size(a,1:9);
for k=1:9
I=[];
data=xlsread('EnergyForecastData.xlsx', a{k}); % importing data

x=data(:,1);   % year
y=data(:,3);  % annual carbon emission
j=data(:,2);  % population
w=data(:,4);  % growth domestic product gdp per capita
t=data(:,5);
s=data(:,6);
n=data(:,7);
m=length(x);
  
%% regression with feature normalization 
x=data(:,1);   % year
y=data(:,3);  % annual carbon emission
j=data(:,2);  % population
w=data(:,4);  % growth domestic product gdp per capita
t=data(:,5);  % fossil fuel % of electricity
s=data(:,6);  % carbon intensity
n=data(:,7);  % Annual CO2 emissions per unit energy

m=length(x);


xnorm = (x- min(x)) / (max(x) - min(x));
ynorm = (y- min(y)) / (max(y) - min(y));

%% effect of model complexity on underfitting or overfitting 

p=0.8;
nrepeats = 10;
maxdegree=3;
errortrain = zeros(maxdegree,1);
errorval = zeros(maxdegree,1);

for ii=1:nrepeats 
idx=randperm(m); 
xtrain = xnorm(idx(1:round(p*m)));
xval = xnorm(idx(round(p*m)+1:end));
ytrain = ynorm(idx(1:round(p*m)));
yval = ynorm(idx(round(p*m)+1:end));

   for i=1:maxdegree 
    pp = polyfit(xtrain, ytrain, i);
    ypredtrain = polyval(pp,xtrain);
    ypredval = polyval(pp, xval);
    errortrain(i) = errortrain(i) + sqrt(mean((ypredtrain-ytrain).^2));
    errorval(i) = errorval(i) + sqrt(mean((ypredval-yval).^2));
   end
end

errortrain = errortrain/nrepeats;
errorval = errorval/nrepeats;
figure; % PLEASE COMPLETE THE PLOTS
plot(1:maxdegree, errortrain); hold on;
plot(1:maxdegree, errorval); hold off
xlabel('Polynomial Degree Compatibility')
ylabel('RMSE(C02 Emissions In tonnes)')
title('Regression Model Degree Evaluation', a{k})
legend('Validated','Trained')

% OPTIMAL POLYNOMIAL DEGREE
[M, I] = min(errorval);
I_val = [I_val I];
M_val = [M_val M];

pp=polyfit(x,y,I);
xfuture=2021:2050;
yfuturepred = polyval(pp,xfuture);
figure;   % PLEASE COMPLETE THE PLOTS
plot(x,y); hold on
plot(xfuture,yfuturepred); hold off
xlabel('Year')
ylabel('CO2 Emissions (tonnes)')
title('Annual C02 Emissions',a(k))
legend('Current','Predicted')

% Final_prediction = [xfuture' yfuturepred'];
% Final_prediction_val = [Final_prediction_val yfuturepred'];    
    
    
end 

% x= [YEAR];
% j= [POPULATION];
% w= [GROWTH-DOMESTIC-PRODUCT];
% y= [ANNUAL-CO2-EMMISSION];

% X=[x j w t s n];
% [idx,weights] = relieff(X,y,3)

XX=[x(11:30,:) j(11:30,:) w(11:30,:) t(11:30,:) s(11:30,:) n(11:30,:) y(11:30,:)];
X= [x(11:30,:) j(11:30,:) w(11:30,:) t(11:30,:) s(11:30,:) n(11:30,:)];
Y=y(11:30,:);
[idx,weights] = relieff(X,Y,6)

corrplot(XX)



