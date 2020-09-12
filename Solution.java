class Point {
    int i, j;
    Point (int row, int col) {
        i = row; j = col;
    }
}

class Solution {    
    /** 09-05-20 **/
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
