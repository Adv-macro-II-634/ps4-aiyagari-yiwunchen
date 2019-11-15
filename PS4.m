clear, clc
tic

alpha=1/3; 
beta = .99; 
sigma = 2; 
delta=.25; %suppose depreciation rate=0.25
rho=0.5; 
sigma_e=0.2; 

% ASSET VECTOR
a_lo = 0; %lower bound of grid points
a_hi = 1000;
%we need to guess a_max and verify later that it does not constitute a
%binding constraint on how many assets households wish to hold.
num_a = 500;
a = linspace(a_lo, a_hi, num_a); % asset (row) vector

%worker's labor efficiency z
m=5;
[z_grid, z_prob] = rouwenhorst(rho,sigma_e,m); 


% INITIAL GUESS FOR K
K_min=0;
K_max=1000;
K_guess = (K_min + K_max) / 2;

% ITERATE OVER K
K_tol=1;
while abs(K_tol)>=.01;
    
    N=ones(1,m)*z_prob(1,:)' %suppose the mass of total labor =1
    K_guess=(K_min+K_max)/2;
    r= alpha*K_guess^(alpha-1)*N^(1-alpha)+(1-delta); %rental rate
    w=(1-alpha)*K_guess^alpha*N^(-alpha); %wage rate
    
   	cons = bsxfun(@minus, a', r*a);
    cons = bsxfun(@plus, cons, permute(z_grid*w, [1 3 2]));
    ret = (cons .^ (1-sigma)) ./ (1 - sigma); % current period utility
    ret(cons<0)=-Inf;
    
    % INITIAL VALUE FUNCTION GUESS
    
    v_guess = zeros(m, num_a);
    
% VALUE FUNCTION ITERATION
v_tol = 1;
while v_tol >.0001;
  % CONSTRUCT TOTAL RETURN FUNCTION
  v_mat = ret + beta * ...
       repmat(permute(z_prob * v_guess, [3 2 1]), [num_a 1 1]);
  
  % CHOOSE HIGHEST VALUE (ASSOCIATED WITH a' CHOICE)
   [vfn, pol_indx] = max(v_mat, [], 2);
   vfn = permute(vfn, [3 1 2]);
   v_tol = abs(max(v_guess(:) - vfn(:)));
 
   v_guess = vfn; %update value functions
        
end;

  % KEEP DECSISION RULE  
    pol_indx=permute(pol_indx, [3 1 2]);
    pol_fn = a(pol_indx);
    
% SET UP INITITAL DISTRIBUTION
Mu = zeros(m,num_a);
Mu(1, 4) = 1; 
%Mu(:)=1/(m*num_a);

% ITERATE OVER DISTRIBUTIONS
% loop over all non-zeros states
mu_tol = 1;
while mu_tol > 1e-08
    [z_ind, a_ind] = find(Mu > 0); % find non-zero indices
    
    MuNew = zeros(size(Mu));
    for ii = 1:length(z_ind)
        apr_ind = pol_indx(z_ind(ii), a_ind(ii)); 
        MuNew(:, apr_ind) = MuNew(:, apr_ind) + ...
            (z_prob(z_ind(ii), :) * Mu(z_ind(ii), a_ind(ii)) )';
    end
    
    mu_tol = max(abs(MuNew(:) - Mu(:)));
    
    Mu = MuNew ;
    
end
   
   % CHECK AGGREGATE DEMAND
   K_tol=sum( pol_fn(:) .* Mu(:) );
if K_tol>0;
  K_min=K_guess;
end
if K_tol < 0;
  K_max=K_guess;
end;


display (['K = ', num2str(K_guess)])
display (['Aggregate desired capital = ', num2str(K_tol)]);
display (['New K_min is ', num2str(K_min), ', new K_max is ', num2str(K_max)]);
display (['New K is ', num2str((K_max + K_min)/2)]);

K_guess = (K_max + K_min)/2 ;

display (' ') ;
end

%% FIND TOTAL WEALTH DISTRIBUTION AND GINI
agg_wealth = sum(Mu,1) * a'; % wealth is asset holdings 
w_z1 = r*a+z_grid(1)*w;
w_z2 = r*a+z_grid(2)*w;
w_z3 = r*a+z_grid(3)*w;
w_z4 = r*a+z_grid(4)*w;
w_z5 = r*a+z_grid(5)*w;
wealth_dist = [[Mu(1,:), Mu(2,:), Mu(3,:), Mu(4,:), Mu(5,:)]; [w_z1, w_z2,w_z3,w_z4,w_z5]]';
[~, ordr] = sort(wealth_dist(:,2), 1);
wealth_dist = wealth_dist(ordr,:);


% see formula on wikipedia for computation of gini in discrete
% distributions:
pct_dist = cumsum( (wealth_dist(:,2) ./ agg_wealth) .* wealth_dist(:,1) );
gini = 1 - sum( ([0; pct_dist(1:end-1)] + pct_dist) .* wealth_dist(:,1) );

display (['Gini coefficient of ', num2str(gini)]);


%plot the policy function:
plot(a,pol_fn)
xlabel('asset') 
ylabel('Policy Function')
title('Policy Function for the m productivity states(z)')

% Plot the Lorenz curve
w= [w_z1 w_z2 w_z3 w_z4 w_z5].*Mu(:)';
% calculate the cumulative share of iwealth
a1=sort(w); 
a2=a1(a1(:,1)>=0);
a3=cumsum(a2);
a4=a3/sum(a2);
% calculate the cumulative share of people from lowest to highest incomes
a5=[1:length(a2)]';
a6=a5/length(a5);

plot(a6,a4);
xlabel("Cumulative share of people from lowest to highest wealth");
ylabel("Cumulative share of wealth");
title("the Lorenz curve");
hold on;