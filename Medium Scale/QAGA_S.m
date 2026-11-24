%% QAGA-S — Q-learning Adaptive GA, Selection-Only
%  Q-learning فقط روش انتخاب والدین را کنترل می‌کند (3 اکشن)
%  پارامترهای mu و gamma ثابت باقی می‌مانند
%  نقش: اندازه‌گیری اثر خالص «انتخاب هوشمند» بدون تداخل سایر پارامترها
%  سوال: آیا فقط انتخاب هوشمند والدین به‌تنهایی کافی است؟
% ---------------------------------------------------------------
clear; close all;

load('Cost_3d_data_50_20.mat',   'Cost');
load('V_cost_3d_data_50_20.mat', 'V_cost');
load('D_P_3d_data_50_20.mat',    'D_P');
load('V_C_3d_data_50_20.mat',    'V_C');

%% ===== پارامترهای GA (ثابت) =====
nVar     = 70;
VarSize  = [1 nVar];
VarMin   = 0;    VarMax = 1;
MaxIt    = 500;
nPop     = 30;
pc       = 0.7;
nc       = 2*round(pc*nPop/2);
gamma    = 0.4;      % ثابت
pm       = 0.3;
nm       = round(pm*nPop);
mu       = 0.1;      % ثابت
beta     = 8;
TournamentSize = 3;

%% ===== پارامترهای Q-learning =====
% فضای حالت: 16 حالت (4 سطح تنوع × 4 مرحله بهینه‌سازی)
% فضای اکشن: 3 (Roulette=1, Tournament=2, Random=3)
nStates  = 16;
nActions = 3;   % <<< کوچک‌ترین فضای اکشن در بین الگوریتم‌ها
alpha_rl = 0.1;   % نرخ یادگیری Q
epsilon  = 0.5;   % نرخ کاوش

states = [1 1; 1 2; 1 3; 1 4;
          2 1; 2 2; 2 3; 2 4;
          3 1; 3 2; 3 3; 3 4;
          4 1; 4 2; 4 3; 4 4];

%% ===== آرایه‌های خروجی =====
nRuns        = 10;
Conv_it      = zeros(1, nRuns);
Best_found   = zeros(1, nRuns);
Elapsed_time = zeros(1, nRuns);

empty_individual.Position = [];
empty_individual.Cost     = [];

for run = 1:nRuns
    fprintf('QAGA-S — Run %d/%d\n', run, nRuns);
    tic;

    %% بازنشانی Q-table برای هر اجرا
    Q = zeros(nStates, nActions);

    UseRoulette   = false;
    UseTournament = false;
    UseRandom     = true;

    state  = 1;
    action = 3;   % شروع با Random

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

        %% --- RL: تصمیم هر 5 تکرار ---
        if rem(it, 5) == 0

            % محاسبه پاداش
            if it > 5
                reward = BestCost(it - 6) - BestCost(it - 1);
                Q(state, action) = Q(state, action) + alpha_rl * (reward - Q(state, action));
            end

            % تعریف حالت جدید
            convergence = (std(Costs(~isnan(Costs))) / mean(Costs(~isnan(Costs)))) * 100;
            s1 = 1 + (convergence >= 0.9) + (convergence >= 1.8) + (convergence >= 2.7);
            s2 = min(4, floor(it / (MaxIt/4)) + 1);
            state = find(ismember(states, [s1 s2], 'rows'));

            % انتخاب اکشن (ε-greedy)
            if rand < epsilon
                action = randi([1, nActions]);
            else
                [~, action] = max(Q(state, :));
            end

            % اعمال اکشن: فقط روش انتخاب تغییر می‌کند
            UseRoulette   = (action == 1);
            UseTournament = (action == 2);
            UseRandom     = (action == 3);
        end

        %% --- GA: Crossover ---
        if UseRoulette
            P = exp(-beta * Costs / WorstCost);
            P = P / sum(P);
        end
        popc = repmat(empty_individual, nc/2, 2);
        for k = 1:nc/2
            if UseRoulette
                i1 = RouletteWheelSelection(P);
                i2 = RouletteWheelSelection(P);
            elseif UseTournament
                i1 = TournamentSelection(pop, TournamentSize);
                i2 = TournamentSelection(pop, TournamentSize);
            else
                i1 = randi([1 nPop]);
                i2 = randi([1 nPop]);
            end
            [popc(k,1).Position, popc(k,2).Position] = ...
                Crossover(pop(i1).Position, pop(i2).Position, gamma, VarMin, VarMax);
            popc(k,1).Cost = ObjectiveFunction(popc(k,1).Position, Cost, V_cost, D_P, V_C);
            popc(k,2).Cost = ObjectiveFunction(popc(k,2).Position, Cost, V_cost, D_P, V_C);
        end
        popc = popc(:);

        %% --- GA: Mutation (پارامتر ثابت) ---
        popm = repmat(empty_individual, nm, 1);
        for k = 1:nm
            p = pop(randi([1 nPop]));
            popm(k).Position = Mutate(p.Position, mu, VarMin, VarMax);
            popm(k).Cost     = ObjectiveFunction(popm(k).Position, Cost, V_cost, D_P, V_C);
        end

        %% --- ادغام، مرتب‌سازی، برش ---
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

%% ===== ذخیره و نمایش =====
save('results_QAGA_S.mat', 'Best_found', 'Conv_it', 'Elapsed_time');
fprintf('\n===== QAGA-S Results =====\n');
fprintf('Best  : %.6f ± %.6f\n', mean(Best_found),   std(Best_found));
fprintf('Conv  : %.1f  ± %.1f\n', mean(Conv_it),      std(Conv_it));
fprintf('Time  : %.2fs ± %.2fs\n', mean(Elapsed_time), std(Elapsed_time));

figure; semilogy(BestCost, 'LineWidth', 2);
xlabel('Iteration'); ylabel('Best Cost'); title('QAGA-S'); grid on;
