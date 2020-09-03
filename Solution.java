class Solution {    
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
}
