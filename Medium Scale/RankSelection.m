function i = RankSelection(pop)

    nPop = numel(pop);
    
    % مرتب‌سازی بر اساس هزینه
    Costs = [pop.Cost];
    [~, SortOrder] = sort(Costs);
    
    % تخصیص رتبه: بالاترین شانس به بهترین فرد
    RankWeights = nPop - (1:nPop) + 1; % یعنی بهترین فرد وزن nPop می‌گیره
    
    % نرمال‌سازی وزن‌ها
    SelectionProb = RankWeights / sum(RankWeights);
    
    % انتخاب با چرخ رولت بر اساس رتبه
    r = rand();
    cumProb = cumsum(SelectionProb);
    i = SortOrder(find(r <= cumProb, 1, 'first'));
    
end
