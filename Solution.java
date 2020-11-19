class Point {
    int i, j;
    Point (int row, int col) {
        i = row; j = col;
    }
}

class Solution {
    /** 11-14-20 **/
    
    // 53. Maximum Subarray

    
    // 139. Word Break
    HashMap<Integer, Boolean> memo = new HashMap<>();    
    
    public boolean wordBreak(String s, List<String> wordDict) {
        HashSet<String> dict = new HashSet<>(wordDict);
        return dfs(s, 0, dict);
    }

    boolean dfs(String s, int index, HashSet<String> dict) {
        if (index == s.length()) {
            return true;
        } 
        if (memo.containsKey(index)) {
            return memo.get(index);
        }
        boolean ans = false;
        for (int cut = index; cut < s.length() && !ans; cut++) {
            // s[index, index + 1, ... cut]
            String word = s.substring(index, cut + 1);
            if (dict.contains(word)) {
                // substring s[cut + 1:]
                if (dfs(s, cut + 1, dict)) {
                    ans = true;
                }
            }        
        }
        memo.put(index, ans);
        return ans;
    }

    // 509. Fibonacci Number
    HashMap<Integer, Integer> memo = new HashMap<>();
    
    public int fib(int N) {
        if (N == 0 || N == 1) return N;
        if (memo.containsKey(N)) return memo.get(N);
        int ans = fib(N - 1) + fib(N - 2);
        memo.put(N, ans);
        return ans;
    }

    /** 10-31-20 **/
    
    // 756. Pyramid Transition Matrix
    public boolean pyramidTransition(String bottom, List<String> allowed) {
        Map<String, List<String>> map = new HashMap<>();
        for (String s : allowed) {
            if (!map.containsKey(s.substring(0, 2))) {
                map.put(s.substring(0, 2), new ArrayList<>());
            }
            map.get(s.substring(0, 2)).add(s.substring(2, 3));
        }
        
        return dfs(bottom, map);
    }
    
    boolean dfs(String bottom, Map<String, List<String>> map) {
        if (bottom.length() == 1) return true;
        
        List<String> nextBottoms = getNextBottoms(bottom, map);
        for (String nextBottom : nextBottoms) {
            if (dfs(nextBottom, map)) {
                return true;
            }
        }
        
        return false;
    }
    
    List<String> getNextBottoms(String bottom, 
                                Map<String, List<String>> map) {
        List<String> nextBottoms = new ArrayList<>();
        for (int i = 0; i < bottom.length() - 1; i++) {
            if (!map.containsKey(bottom.substring(i, i + 2))) {
                return nextBottoms;
            }
        }
        helper(bottom, 0, "", nextBottoms, map);
        return nextBottoms;
    }
    
    void helper(String bottom, int index, String path,
                List<String> nextBottoms, 
                Map<String, List<String>> map) {
        if (path.length() == bottom.length() - 1) {
            nextBottoms.add(path);
            return;
        }
        
        for (String ch : map.get(bottom.substring(index, index + 2))) {
            helper(bottom, index + 1, path + ch, nextBottoms, map);            
        }
    }

    // 212. Word Search II
    class TrieNode {
        String word = "";
        Map<Character, TrieNode> children = new HashMap<>();
    }

    class Trie {
        TrieNode root = new TrieNode();
        void add(String word) {
            TrieNode cur = root;
            for (int i = 0; i < word.length(); i++) {
                if (!cur.children.containsKey(word.charAt(i))) {
                    cur.children.put(word.charAt(i), new TrieNode());
                }
                cur = cur.children.get(word.charAt(i));
            }
            cur.word = word;
        }
    }
    
    Trie trie = null;
    int[][] delta = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    
    public List<String> wordSearchII(char[][] board, List<String> words) {
        trie = new Trie();
        Set<String> visited = new HashSet<>();
        for (String word : words) {
            trie.add(word);
        }
        
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                dfs(board, i, j, trie.root, visited);
            }
        }
        
        List<String> ans = new ArrayList<>();
        for (String word: visited) {
            ans.add(word);
        }
        return ans;
    }
    
    private void dfs(char[][] board, int i, int j, TrieNode node, Set<String> visited) {
        if (!node.word.equals("")) {
            visited.add(node.word);
        }
        
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length) {
            return;
        }
        
        if (!node.children.containsKey(board[i][j])) {
            return;
        }
        
        char original = board[i][j];
        board[i][j] = '#';
        for (int d = 0; d < delta.length; d++) {
            int ni = i + delta[d][0], nj = j + delta[d][1];
            dfs(board, ni, nj, node.children.get(original), visited);
        }
        board[i][j] = original;
    }

    // 3. Longest Substring Without Repeating Characters
    public int lengthOfLongestSubstring(String s) {
        HashMap<Character, Integer> count = new HashMap<>();
        
        int ans = 0, left = 0, right = 0;
        
        while (right < s.length()) {
            // [left, right)
            char cur = s.charAt(right);
            count.put(cur, count.getOrDefault(cur, 0) + 1);
            // [left, right]
            while (count.get(cur) > 1) { // while invalid
                char pre = s.charAt(left);
                count.put(pre, count.get(pre) - 1);
                left++;
            }
            // [left, right] valid
            ans = Math.max(ans, right - left + 1);
            right++;
            // [left, right)
        }
        
        return ans;
    }    

    /** 10-17-20 **/

    // 212. Word Search II
    class TrieNode {
        String word = "";
        Map<Character, TrieNode> children = new HashMap<>();
    }
    
    class Trie {
        TrieNode root = null;
        Trie() {
            root = new TrieNode();
        }
        void add(String word) {
            TrieNode cur = root;
            for (int i = 0; i < word.length(); i++) {
                if (!cur.children.containsKey(word.charAt(i))) {
                    cur.children.put(word.charAt(i), new TrieNode());
                }
                cur = cur.children.get(word.charAt(i));
            }
            cur.word = word;
        }
        int isPrefix(String word) {
            TrieNode cur = root;
            for (int i = 0; i < word.length(); i++) {
                if (!cur.children.containsKey(word.charAt(i)))  {
                    return 0;
                }
                cur = cur.children.get(word.charAt(i));
            }
            if(cur.word.equals("")) return 1; // 仅仅是一个前缀
            return 2; // 是一个单词
        }
    }
    
    Trie trie = null;
    int[][] delta = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    
    public List<String> findWords(char[][] board, String[] words) {
        trie = new Trie();
        Set<String> visited = new HashSet<>();
        for (String word : words) {
            trie.add(word);
        }
        
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                dfs(board, i, j, "", visited);
            }
        }
        
        List<String> ans = new ArrayList<>();
        for (String word: visited) {
            ans.add(word);
        }
        return ans;
    }
    
    private void dfs(char[][] board, int i, int j, String path, Set<String> visited) {
        int flag = trie.isPrefix(path);
        if (flag == 0) {
            return;
        }
        if (flag == 2) {
            visited.add(path);
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length) {
            return;
        }
        char original = board[i][j];
        board[i][j] = '#';
        for (int d = 0; d < delta.length; d++) {
            int ni = i + delta[d][0], nj = j + delta[d][1];
            dfs(board, ni, nj, path + original, visited);
        }        
        board[i][j] = original;        
    }    
    
    // 425. Word Squares
    class Node {
        public Map<Character, Node> children = new HashMap<>();
        public String word = "";
        public List<Integer> words = new ArrayList<>();
    }
    
    class Trie {
        public Node root = new Node();
        public void add(String word, int id) {
            Node cur = root;
            root.words.add(id);
            for (int i = 0; i < word.length(); i++) {
                if (!cur.children.containsKey(word.charAt(i))) {
                    cur.children.put(word.charAt(i), new Node());
                }
                cur = cur.children.get(word.charAt(i));
                cur.words.add(id);
            }
            cur.word = word;
        }
    }
    
    Trie trie = new Trie();
    
    public List<List<String>> wordSquares(String[] words) {
        List<List<String>> ans = new ArrayList<>();
        if (words == null || words.length == 0) {
            return ans;
        }
        
        for (int i = 0; i < words.length; i++) {
            trie.add(words[i], i);
        }
        List<String> path = new ArrayList<>();
        dfs(words, 0, words[0].length(), path, ans);
        return ans;
    }
    
    private void dfs(String[] words, int row, int target, List<String> path, List<List<String>> ans) {
        if (row == target) {
            ans.add(new ArrayList<>(path));
            return;
        }
        
        Node cur = trie.root;
        for (int i = 0; i < path.size(); i++) {
            if (!cur.children.containsKey(path.get(i).charAt(row))) {
                return;
            }
            cur  = cur.children.get(path.get(i).charAt(row));
        }
        for (Integer index : cur.words) {
            path.add(words[index]);
            dfs(words, row + 1, target, path, ans);
            path.remove(path.size() - 1);
        }
    }

    // 1120. Maximum Average Subtree
    public TreeNode findSubtree2(TreeNode root) {
        return helper(root).maxAvgSubTree;
    }
    
    private Result helper(TreeNode node) {
        if (node == null) {
            return new Result(0, 0, 0, null);
        }
        
        Result left = helper(node.left);
        Result right = helper(node.right);
        
        int sum = left.sum + right.sum + node.val;
        int size = left.size + right.size + 1;
        double currAvg = (double) sum / size;
     
        Result result = new Result(sum, size, currAvg, node);
        
        if (left.maxAvgSubTree != null && result.maxAvg <= left.maxAvg) {
            result.maxAvg = left.maxAvg;
            result.maxAvgSubTree = left.maxAvgSubTree;
        }
        
        if (right.maxAvgSubTree != null && result.maxAvg <= right.maxAvg) {
            result.maxAvg = right.maxAvg;
            result.maxAvgSubTree = right.maxAvgSubTree;
        }
        
        return result;
    }
    
    class Result {
        int sum; 
        int size;
        double maxAvg;
        TreeNode maxAvgSubTree;
        
        Result(int sum, int size, double maxAvg, TreeNode maxAvgSubTree) {
            this.sum = sum;
            this.size = size;
            this.maxAvg = maxAvg;
            this.maxAvgSubTree = maxAvgSubTree;
        }
    }    
    
    /** 10-10-20 **/
    
    // 236. Lowest Common Ancestor of a Binary Tree
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        return helper(root, p, q).target;
    }
    
    private Result helper(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null) {
            return new Result(null, false, false);
        }
        
        Result left = helper(root.left, p, q);
        Result right = helper(root.right, p, q);
        
        if (left.target != null) {
            return left;
        }
        
        if (right.target != null) {
            return right;
        }
        
        boolean foundP = (root == p || left.foundP || right.foundP);
        boolean foundQ = (root == q || left.foundQ || right.foundQ);
        
        if (foundP && foundQ) {
            return new Result(root, true, true);
        }
        return new Result(null, foundP, foundQ);
    }
    
    class Result {
        public TreeNode target;
        public boolean foundP;
        public boolean foundQ;
        public Result(TreeNode target, boolean foundP, boolean foundQ) {
            this.target = target;
            this.foundP = foundP;
            this.foundQ = foundQ;
        }
    }
    
    // 425. Word Squares
    public List<List<String>> wordSquares(String[] words) {
        List<List<String>> ans = new ArrayList<>();
        List<String> path = new ArrayList<>();
        dfs(words, 0, words[0].length(), path, ans);
        return ans;
    }
    
    void dfs(String[] words, int row, int target, List<String> path, List<List<String>> ans) {
        if (row == target) {
            ans.add(new ArrayList<>(path));
            return;
        }
        
        for (int i = 0; i < words.length; i++) {
            if (!check(path, row, words[i])) continue;
            path.add(words[i]);
            dfs(words, row + 1, target, path, ans);
            path.remove(path.size() - 1);
        }
    }
    
    boolean check(List<String> path, int row, String word) {
        for (int i = 0; i < path.size(); i++) {
            if (path.get(i).charAt(row) != word.charAt(i)) return false;
        }
        return true;
    }    
    
    // 51. N-Queens
    public List<List<String>> solveNQueens(int n) {
        List<List<String>> res = new ArrayList<>();
        dfs(n, new ArrayList<Integer>(), res);
        return res;
    }
    
    void dfs(int n, List<Integer> position, List<List<String>> result) {
        if (position.size() == n) {
            result.add(build(position));
            return;
        }
        for (int col = 0; col < n; col++) {
            if (check(position, col)) {
                position.add(col);
                dfs(n, position, result);
                position.remove(position.size() - 1);
            }
        }
    }
    
    List<String> build(List<Integer> position) {
        int n = position.size();
        List<String> board = new ArrayList<>();
        for (int row = 0; row < n; row++) {
            String line = "";
            for (int col = 0; col < n; col++) {
                if (position.get(row) == col) {
                    line += "Q";
                } else {
                    line += ".";
                }
            }
            board.add(line);
        }
        return board;
    }
    
    boolean check(List<Integer> path, int col) {
        int row = path.size();
        
        for (int i = 0; i < path.size(); i++) {
            if (path.get(i) == col) return false;
        }
        
        for (int i = 0; i < path.size(); i++) {
            if (i - path.get(i) == row - col) return false;
        }
        
        for (int i = 0; i < path.size(); i++) {
            if (i + path.get(i) == row + col) return false;
        }
        
        return true;
    }
    
    /** 10-03-20 **/
    public TreeNode lcaDeepestLeaves(TreeNode root) {
        return helper(root).target;
    }
    
    private Result helper(TreeNode root) {
        if (root == null) {
            return new Result(null, 0);
        }
        
        Result left = helper(root.left);
        Result right = helper(root.right);
        
        if (left.depth == right.depth) {
            return new Result(root, left.depth + 1);
        } else if (left.depth < right.depth) {
            return new Result(right.target, right.depth + 1);
        } else {
            return new Result(left.target, left.depth + 1);
        }        
    }
    
    class Result {
        TreeNode target;
        int depth;
        Result(TreeNode target, int depth) {
            this.target = target;
            this.depth = depth;
        }
    }
    /** 09-26-20 **/


    /** 09-12-20 **/
    
    // 17. Letter Combinations of a Phone Number
    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<>();
        
        if (digits == null || digits.isEmpty()) {
            return result;
        }
        
        Map<Character, char[]> map = new HashMap<>();
        map.put('2', new char[] { 'a', 'b', 'c' });
        map.put('3', new char[] { 'd', 'e', 'f' });
        map.put('4', new char[] { 'g', 'h', 'i' });
        map.put('5', new char[] { 'j', 'k', 'l' });
        map.put('6', new char[] { 'm', 'n', 'o' });
        map.put('7', new char[] { 'p', 'q', 'r', 's' });
        map.put('8', new char[] { 't', 'u', 'v'});
        map.put('9', new char[] { 'w', 'x', 'y', 'z' });
        
        dfs(map, digits, 0, new StringBuilder(), result);
        
        return result;
    }

    private void dfs(Map<Character, char[]> map, String digits, int index, StringBuilder sb, List<String> result) {
        if (index == digits.length()) {
            result.add(sb.toString());
            return;
        }
        
        char[] str = map.get(digits.charAt(index));
        for (char c : str) {
            sb.append(c);
            dfs(map, digits, index + 1, sb, result);
            sb.setLength(sb.length() - 1);
        }
    }

    // There are N courses, labelled from 1 to N.
    public int minimumSemesters(int N, int[][] relations) {
        
        HashMap<Integer, List<Integer>> nexts = new HashMap<>();
        int[] indegree = new int[N + 1];
        
        for(int[] relation: relations) {
            if(!nexts.containsKey(relation[0])) {
                nexts.put(relation[0], new ArrayList<>());
            }
            nexts.get(relation[0]).add(relation[1]);
            indegree[relation[1]] ++;
        }
        
        Queue<Integer> queue = new LinkedList<>();
        for(int node = 1; node <= N; node ++) {
            if(indegree[node] == 0) queue.offer(node);
        }
        
        // a d
        
        int semester = 0;
        int finished = 0;
        while(queue.size() > 0) {
            int size = queue.size();
            for(int i = 0; i < size; i ++) {
                int cur = queue.poll();
                finished ++;
                for(int next: nexts.getOrDefault(cur, new ArrayList<>())) {
                    indegree[next] --;
                    if(indegree[next] == 0) {
                        queue.offer(next);
                    }
                }
            }
            semester ++;
        }
        if(finished != N) return -1;
        return semester;
    }

    public List<List<Integer>> permute(int[] nums) {
        boolean[] used = new boolean[nums.length];
        List<List<Integer>> result = new ArrayList<>();
        dfs(nums, used, new ArrayList<>(), result);
        return result;
    }
    
    void dfs(int[] nums, boolean[] used, List<Integer> path, List<List<Integer>> result) {
        if (path.size() == nums.length) {
            result.add(new ArrayList<>(path));
        }
        
        for (int i = 0; i < nums.length; i++) {
            if (used[i]) continue;
            path.add(nums[i]);
            used[i] = true;
            dfs(nums, used, path, result);
            path.remove(path.size() - 1);
            used[i] = false;
        }
    }
    
    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int[] indegree = new int[numCourses];// indegree[label] = count;
        List<List<Integer>> nexts = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            nexts.add(new ArrayList<>());
        }
        //nexts.get(label) --> List of label's neighbors
        for (int[] edge : prerequisites) {
            int from = edge[1], to = edge[0];
            nexts.get(from).add(to);
            indegree[to]++;
        }
        // finish graph construct directed graph
        
        Queue<Integer> queue = new LinkedList<>();
        
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) queue.offer(i);
        }
        
        int[] ans = new int[numCourses];
        int index = 0;
        
        while (queue.size() > 0) {
            int cur = queue.poll();
            ans[index++] = cur;
            for (int next : nexts.get(cur)) {
                indegree[next]--;
                if (indegree[next] == 0) {
                    queue.offer(next);
                }
            }
        }
        
        return index == numCourses ? ans : new int[0]; // check cycle
    }
    
    /** 09-05-20 **/
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Set<String> dict = new HashSet<>(wordList);
        if (!dict.contains(endWord)) {
            return 0;
        }
        Queue<String> queue = new ArrayDeque<>();
        queue.offer(beginWord);
        
        int level = 1;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                String word = queue.poll();
                if (Objects.equals(word, endWord)) {
                    return level;
                }
                int len = word.length();
                for (int j = 0; j < len; j++) {
                    char[] array = word.toCharArray();
                    for (int k = 0; k < 26; k++) {
                        array[j] = (char)('a' + k);
                        String newWord = new String(array);
                        if (dict.contains(newWord)) {
                            queue.offer(newWord);
                            dict.remove(newWord);
                        }
                    }
                }
            }
            level++;
        }
        
        return 0;
    }
    
    public void solve(char[][] board) {
        if (board.length == 0 || board[0].length == 0) return;
        int m = board.length, n = board[0].length;
        
        Queue<Point> queue = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if ((i == 0 || i == m - 1 || j == 0 || j == n -1) && board[i][j] == 'O') {
                    board[i][j] = 'V';
                    queue.offer(new Point(i, j));
                } 
            }
        }
    
        bfs(board, queue);
    
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'V') board[i][j] = 'O';
                else board[i][j] = 'X';
            }
        }
    }
    
    int[][] delta = new int[][] {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    
    void bfs(char[][] board, Queue<Point> queue) {
        while (queue.size() > 0) {
            int size = queue.size();
            for (int count = 0; count < size; count++) {
                Point cur = queue.poll();
                for (int d = 0; d < 4; d++) {
                    int ni = cur.i + delta[d][0], nj = cur.j + delta[d][1];
                    if (ni < 0 || ni >= board.length || nj < 0 || nj >= board[0].length) continue;
                    if (board[ni][nj] != 'O') continue;
                    board[ni][nj] = 'V';
                    queue.offer(new Point(ni, nj));
                }
            }
        }
    }

    public int shortestDistance(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
        
        int m = grid.length, n = grid[0].length;
        int[][] totalDistance = new int[m][n];
        int[][] canReach = new int[m][n]; //canReach[i][j] = k;
        int count = 0;
        
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    bfs(grid, i, j, canReach, totalDistance);
                    count ++;
                }
            }
        }
        
        // count : the number of building
        
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if(grid[i][j] == 0 && canReach[i][j] == count) {
                    res = Math.min(res, totalDistance[i][j]);
                }
            }
        }
        return res == Integer.MAX_VALUE ? -1 : res;
    }
    
    private void bfs(int[][] grid, int x, int y, int[][] canReach, int[][] totalDistance) {
        int res = Integer.MAX_VALUE, m = grid.length, n = grid[0].length;;
        
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(x * n + y);
        boolean[][] visited = new boolean[m][n];
        visited[x][y] = true;
        
        int curDis = 0;
        int[] dirs = {-1, 0, 1, 0, -1};
        
        while (!queue.isEmpty()) {
            int l = queue.size();
            curDis++;
            while (l-- != 0) {
                int t = queue.poll();
                x = t / n;
                y = t % n;
                
                canReach[x][y] ++;
                
                for (int i = 0; i < 4; ++i) {
                    int _x = x + dirs[i], _y = y + dirs[i + 1];
                    if (_x >= 0 && _x < m && _y >= 0 && _y < n && grid[_x][_y] == 0 && !visited[_x][_y]) {
                        queue.offer(_x * n + _y);
                        visited[_x][_y] = true;
                        totalDistance[_x][_y] += curDis;
                        
                    }
                }
            }
        }
    }
    
    /** 08-29-20 **/
    int[][] delta = new int[][] {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1},{-1, -1}, {-1, 1}, {1, -1}}; 
    public int shortestPathBinaryMatrix(int[][] grid) {
        if (grid.length == 0 || grid[0].length == 0) return -1;
        int m = grid.length, n = grid[0].length;
        if (grid[0][0] == 1 || grid[m - 1][n - 1] == 1) return -1;
        
        Queue<int[]> queue = new LinkedList<>();
        boolean[][] visited = new boolean[m][n];
        
        queue.offer(new int[]{0, 0});
        visited[0][0] = true;
        
        int step = 0;
        
        while (queue.size() > 0) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] cur = queue.poll();
                if (cur[0] == m - 1 && cur[1] == n - 1) return step + 1;
                for (int d = 0; d < 8; d++) {
                    int ni = cur[0] + delta[d][0], nj = cur[1] + delta[d][1];
                    if (ni < 0 || ni >= m || nj < 0 || nj >= n || visited[ni][nj] || grid[ni][nj] == 1) {
                        continue;
                    }
                    queue.offer(new int[]{ni, nj});
                    visited[ni][nj] = true;
                }
            }
            step++;
        }
        
        return -1;
    }
    
    /** 08-22-20 **/    
    public int findLast(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left + 1 < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] <= target) {
                left = mid;
            } else {
                right = mid;
            }
        }
        if (nums[right] == target) return right;
        if (nums[left] == target) return left;
        return -1;
    }
    
    /** 08-16-20 **/
    public List<List<Integer>> subsets(int[] nums) {
        if (nums.length == 0) return new ArrayList<>();

        List<Integer> path = new ArrayList<>();
        List<List<Integer>> result = new ArrayList<>();
        
        subsets(nums, 0, path, result);
        
        return result;
    }
    
    private void subsets(int[] nums, 
                         int index, 
                         List<Integer> path,
                         List<List<Integer>> result) {
        if (index == nums.length) {
            result.add(new ArrayList<>(path));
            return;
        }
        
        subsets(nums, index + 1, path, result);
        
        path.add(nums[index]);
        subsets(nums, index + 1, path, result);
        path.remove(path.size() - 1);
    }    
}
