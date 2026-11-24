%% BGA — Baseline Genetic Algorithm
%  پارامترهای کاملاً ثابت، بدون هیچ مکانیزم تطبیق پارامتر
%  نقش: خط مبنا (lower bound) برای مقایسه با سایر الگوریتم‌ها
% ---------------------------------------------------------------
clc; clear; close all;

load('Cost_3d_data_10_4.mat',   'Cost');
load('V_cost_3d_data_10_4.mat', 'V_cost');
load('D_P_3d_data_10_4.mat',    'D_P');
load('V_C_3d_data_10_4.mat',    'V_C');

%% ===== پارامترهای ثابت GA =====
nVar     = 14;
VarSize  = [1 nVar];
VarMin   = 0;    VarMax = 1;

MaxIt    = 500;
nPop     = 30;

pc       = 0.7;
nc       = 2*round(pc*nPop/2);
gamma    = 0.4;      % extra-range factor for arithmetic crossover

pm       = 0.3;
nm       = round(pm*nPop);
mu       = 0.1;      % mutation step size

% روش انتخاب ثابت (Random)
UseRouletteWheelSelection = false;
UseTournamentSelection    = false;
UseRandomSelection        = true;

beta           = 8;
TournamentSize = 3;

%% ===== آرایه‌های خروجی (50 اجرای مستقل) =====
nRuns        = 10;
Conv_it      = zeros(1, nRuns);
Best_found   = zeros(1, nRuns);
Elapsed_time = zeros(1, nRuns);

empty_individual.Position = [];
empty_individual.Cost     = [];

for run = 1:nRuns
    fprintf('BGA — Run %d/%d\n', run, nRuns);
    tic;

    %% مقداردهی اولیه جمعیت
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

    %% حلقه اصلی GA
    for it = 1:MaxIt

        % احتمال انتخاب چرخ رولت (فقط اگر فعال باشد)
        if UseRouletteWheelSelection
            P = exp(-beta * Costs / WorstCost);
            P = P / sum(P);
        end

        % --- Crossover ---
        popc = repmat(empty_individual, nc/2, 2);
        for k = 1:nc/2
            if UseRouletteWheelSelection
                i1 = RouletteWheelSelection(P);
                i2 = RouletteWheelSelection(P);
            elseif UseTournamentSelection
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

        % --- Mutation ---
        popm = repmat(empty_individual, nm, 1);
        for k = 1:nm
            p = pop(randi([1 nPop]));
            popm(k).Position = Mutate(p.Position, mu, VarMin, VarMax);
            popm(k).Cost     = ObjectiveFunction(popm(k).Position, Cost, V_cost, D_P, V_C);
        end

        % --- ادغام، مرتب‌سازی، برش ---
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

%% ===== ذخیره و نمایش نتایج =====
save('results_BGA.mat', 'Best_found', 'Conv_it', 'Elapsed_time');
fprintf('\n===== BGA Results =====\n');
fprintf('Best  : %.6f ± %.6f\n', mean(Best_found),   std(Best_found));
fprintf('Conv  : %.1f  ± %.1f\n', mean(Conv_it),      std(Conv_it));
fprintf('Time  : %.2fs ± %.2fs\n', mean(Elapsed_time), std(Elapsed_time));

figure; semilogy(BestCost, 'LineWidth', 2);
xlabel('Iteration'); ylabel('Best Cost'); title('BGA'); grid on;
