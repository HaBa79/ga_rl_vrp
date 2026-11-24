%% QAGA-R — Q-learning Adaptive GA, Operator-Rate Control
%  Q-learning نرخ جهش (mu) و extra-range کراس‌اور (gamma) را کنترل می‌کند
%  روش انتخاب ثابت است (Random)
%  نقش: مقایسه «کنترل شدت اپراتور» در برابر «کنترل نوع انتخاب» (QAGA-S)
%  سوال: مهم‌تر است بدانیم «با چه شدتی» یا «با چه روشی» عملیات ژنتیک انجام دهیم؟
% ---------------------------------------------------------------
clear; close all;

load('Cost_3d_data_50_20.mat',   'Cost');
load('V_cost_3d_data_50_20.mat', 'V_cost');
load('D_P_3d_data_50_20.mat',    'D_P');
load('V_C_3d_data_50_20.mat',    'V_C');

%% ===== پارامترهای GA =====
nVar     = 70;
VarSize  = [1 nVar];
VarMin   = 0;    VarMax = 1;
MaxIt    = 500;
nPop     = 30;
pc       = 0.7;
nc       = 2*round(pc*nPop/2);
pm       = 0.3;
nm       = round(pm*nPop);

% روش انتخاب ثابت (Random) — هیچ‌وقت توسط RL تغییر نمی‌کند
UseRouletteWheelSelection = false;
UseTournamentSelection    = false;
UseRandomSelection        = true;
beta     = 8;
TournamentSize = 3;

%% ===== پارامترهای Q-learning =====
% فضای حالت: 16 (4×4 مثل قبل)
% فضای اکشن: 9
%   اکشن = (جهت mu) × (جهت gamma) × (اندازه گام)
%   3 جهت mu: {کاهش، ثابت، افزایش}
%   3 جهت gamma: {کاهش، ثابت، افزایش}
%   → 3×3 = 9 اکشن
%   جدول اکشن:  action  mu_dir  gamma_dir
%                 1      -1       -1
%                 2      -1        0
%                 3      -1       +1
%                 4       0       -1
%                 5       0        0   (بدون تغییر)
%                 6       0       +1
%                 7      +1       -1
%                 8      +1        0
%                 9      +1       +1
nStates  = 16;
nActions = 9;
alpha_rl = 0.1;
epsilon  = 0.5;
step     = 0.05;   % گام ثابت تغییر پارامتر

states = [1 1; 1 2; 1 3; 1 4;
          2 1; 2 2; 2 3; 2 4;
          3 1; 3 2; 3 3; 3 4;
          4 1; 4 2; 4 3; 4 4];

% جدول نگاشت اکشن به (تغییر mu، تغییر gamma)
action_table = [-1 -1; -1  0; -1 +1;
                 0 -1;  0  0;  0 +1;
                +1 -1; +1  0; +1 +1];

%% ===== آرایه‌های خروجی =====
nRuns        = 10;
Conv_it      = zeros(1, nRuns);
Best_found   = zeros(1, nRuns);
Elapsed_time = zeros(1, nRuns);

empty_individual.Position = [];
empty_individual.Cost     = [];

for run = 1:nRuns
    fprintf('QAGA-R — Run %d/%d\n', run, nRuns);
    tic;

    Q      = zeros(nStates, nActions);
    mu_cur    = 0.1;
    gamma_cur = 0.4;
    state  = 1;
    action = 5;   % شروع با «بدون تغییر»

    pop = repmat(empty_individual, nPop, 1);
    for i = 1:nPop
        pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
        pop(i).Cost     = ObjectiveFunction(pop(i).Position, Cost, V_cost, D_P, V_C);
    end
    Costs = [pop.Cost];
    [Costs, idx] = sort(Costs);
    pop = pop(idx);
    BestCost  = zeros(MaxIt, 1);
    WorstCost = pop(end).Cost;

    for it = 1:MaxIt

        %% --- RL: تصمیم ---
        if rem(it, 5) == 0
            if it > 5
                reward = BestCost(it - 6) - BestCost(it - 1);
                Q(state, action) = Q(state, action) + alpha_rl * (reward - Q(state, action));
            end

            convergence = (std(Costs(~isnan(Costs))) / mean(Costs(~isnan(Costs)))) * 100;
            s1    = 1 + (convergence >= 0.9) + (convergence >= 1.8) + (convergence >= 2.7);
            s2    = min(4, floor(it / (MaxIt/4)) + 1);
            state = find(ismember(states, [s1 s2], 'rows'));

            if rand < epsilon
                action = randi([1, nActions]);
            else
                [~, action] = max(Q(state, :));
            end

            % اعمال اکشن: تغییر mu و gamma (روش انتخاب تغییر نمی‌کند)
            delta_mu    = action_table(action, 1) * step;
            delta_gamma = action_table(action, 2) * step;
            mu_cur    = max(0.01, min(1, mu_cur    + delta_mu));
            gamma_cur = max(0.01, min(1, gamma_cur + delta_gamma));
        end

        %% --- GA: Crossover ---
        popc = repmat(empty_individual, nc/2, 2);
        for k = 1:nc/2
            i1 = randi([1 nPop]);
            i2 = randi([1 nPop]);
            [popc(k,1).Position, popc(k,2).Position] = ...
                Crossover(pop(i1).Position, pop(i2).Position, gamma_cur, VarMin, VarMax);
            popc(k,1).Cost = ObjectiveFunction(popc(k,1).Position, Cost, V_cost, D_P, V_C);
            popc(k,2).Cost = ObjectiveFunction(popc(k,2).Position, Cost, V_cost, D_P, V_C);
        end
        popc = popc(:);

        %% --- GA: Mutation ---
        popm = repmat(empty_individual, nm, 1);
        for k = 1:nm
            p = pop(randi([1 nPop]));
            popm(k).Position = Mutate(p.Position, mu_cur, VarMin, VarMax);
            popm(k).Cost     = ObjectiveFunction(popm(k).Position, Cost, V_cost, D_P, V_C);
        end

        pop   = [pop; popc; popm];
        Costs = [pop.Cost];
        [Costs, idx] = sort(Costs);
        pop       = pop(idx);
        WorstCost = max(WorstCost, pop(end).Cost);
        pop       = pop(1:nPop);
        Costs     = Costs(1:nPop);
        BestCost(it) = pop(1).Cost;
    end

    Elapsed_time(run) = toc;
    Best_found(run)   = BestCost(end);
    tmp               = find(BestCost == BestCost(end), 1, 'first');
    Conv_it(run)      = tmp;
end

save('results_QAGA_R.mat', 'Best_found', 'Conv_it', 'Elapsed_time');
fprintf('\n===== QAGA-R Results =====\n');
fprintf('Best  : %.6f ± %.6f\n', mean(Best_found),   std(Best_found));
fprintf('Conv  : %.1f  ± %.1f\n', mean(Conv_it),      std(Conv_it));
fprintf('Time  : %.2fs ± %.2fs\n', mean(Elapsed_time), std(Elapsed_time));

figure; semilogy(BestCost, 'LineWidth', 2);
xlabel('Iteration'); ylabel('Best Cost'); title('QAGA-R'); grid on;
