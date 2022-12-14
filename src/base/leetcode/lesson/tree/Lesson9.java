package lesson.tree;

import java.util.*;

/**
 * 省份数量
 * 有 n 个城市，其中一些彼此相连，另一些没有相连。如果城市 a 与城市 b 直接相连，且城市 b 与城市 c
 * 直接相连，那么城市 a 与城市 c 间接相连。
 * 省份 是一组直接或间接相连的城市，组内不含其他没有相连的城市。
 * 给你一个 n x n 的矩阵 isConnected ，其中 isConnected[i][j] = 1 表示第 i 个城市和第 j 个城市直接相
 * 连，而 isConnected[i][j] = 0 表示二者不直接相连。
 * 返回矩阵中 省份 的数量。
 * 亲戚问题、朋友圈问题
 */
public class Lesson9 {

    /**
     * 解法1：深度优先
     * 获取一个城市，通过递归找到离该城市最远的城市，标记为已访问，然后逐个向内进行标记
     */
    public static int findCircleNum(int[][] isConnected) {
        int provinces = isConnected.length;
        // 标记是否被访问
        boolean[] visited = new boolean[provinces];
        int circles = 0;
        // 每个未标记的节点标记其所有邻居
        for (int i = 0; i < provinces; i++) {
            if (!visited[i]) {
                dfs(isConnected, visited, provinces, i);
                circles++;
            }
        }
        return circles;
    }

    /**
     * 深度搜索：迭代标记i节点所有关联节点
     */
    public static void dfs(int[][] isConnected, boolean[] visited, int provinces, int i) {
        for (int j = 0; j < provinces; j++) {
            if (isConnected[i][j] == 1 && !visited[j]) {
                visited[j] = true;
                dfs(isConnected, visited, provinces, j);
            }
        }
    }


    /**
     * 解法2：广度优先
     * 获取一个城市，先标记与该城市直连的城市(最近的)，然后逐步向外扩散寻找
     */
    public static int bfs(int[][] isConnected) {
        int provinces = isConnected.length;
        boolean[] visited = new boolean[provinces];
        int circles = 0;
        List<Integer> neighbors = new ArrayList();
        for (int i = 0; i < provinces; i++) {
            if (!visited[i]) {
                neighbors.add(i);
                while (!neighbors.isEmpty()) {
                    neighbors = nextLevel(neighbors, isConnected, visited);
                    circles++;
                }
            }
        }
        return circles;
    }

    /**
     * 没有出现叶子节点时返回全部下一层，否则返回空
     */
    public static List<Integer> nextLevel(List<Integer> neighbors, int[][] isConnected, boolean[] visited) {
        List<Integer> newSubs = new ArrayList();
        for (Integer j : neighbors) {
            visited[j] = true;
            for (int k = 0; k < visited.length; k++) {
                if (isConnected[j][k] == 1 && !visited[k]) {
                    newSubs.add(k);
                }
            }
        }
        return newSubs;
    }

    /**
     * 解法3：并查集
     * 将每个城市看成一个节点，如果两个城市相连，则建立树关系，选出其中一个为head，
     * 如果两个树中的节点也相连，则将其中一个head设置为另一个树的head
     * 两个方法 ：一个寻找head节点，一个合并树
     */
    public static int mergeFind(int[][] isConnected) {
        int provinces = isConnected.length;
        int[] head = new int[provinces];
        int[] level = new int[provinces];
        for (int i = 0; i < provinces; i++) {
            head[i] = i;
            level[i] = 1;
        }
        for (int i = 0; i < provinces; i++) {
            for (int j = i + 1; j < provinces; j++) {
                if (isConnected[i][j] == 1) {
                    // 逐个节点合并树
                    merge(i, j, head, level);
                }
            }
        }
        int count = 0;
        // 找出所有的head
        for (int i = 0; i < provinces; i++) {
            if (head[i] == i) {
                count++;
            }
        }
        return count;
    }

    // 查找head节点
    public static int find(int x, int[] arr) {
        int head = arr[x];
        if ( x == head)
            return head;
        else
            // 逐步查找上一节点，直到根节点
            head = find(head, arr);
        return head;
    }

    public static void merge(int x, int y, int[] arr, int[] level) {
        // 各自的节点合并
        int head1 = find(x, arr);
        int head2 = find(y, arr);

        // 同一根节点直接返回
        if (head1 == head2) {
            return;
        }

        // 浅的树合并到深得树上且把浅的树深度累加
        if (level[head1] <= level[head2]) {
            arr[head1] = head2;
            // 深度加1
            level[head1]++;
        } else {
            arr[head2] = head1;
            // 深度加1
            level[head2]++;
        }
    }



    /**
     * 解法4：集合标记法
     */
    public static int map(int[][] isConnected) {
        Map<Integer, Set<Integer>> map = new HashMap();
        int size = isConnected.length;

        // 1、邻接矩阵
        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                if (isConnected[i][j] == 1) {
                   if(!map.containsKey(i)){
                       map.put(i, new HashSet<>());
                   }
                    if(!map.containsKey(j)){
                        map.put(j, new HashSet<>());
                    }
                    map.get(i).add(j);
                    map.get(j).add(i);
                }
            }
        }

        // 2、标记所有树的head
        boolean[] visited = new boolean[size];
        boolean[] isHead = new boolean[size];
        for (int i = 0; i < size; i++) {
            if (!visited[i]) {
                isHead[i] = true;
                // 独立节点
                if(!map.containsKey(i)){
                    continue;
                }
                Set<Integer> newSubs = new HashSet();
                newSubs.add(i);
                while (!newSubs.isEmpty()) {
                    newSubs = nextLevel(map, newSubs, visited);
                }
            }
        }
        int circles = 0;
        for(boolean b:isHead){
            if(b){
                circles ++;
            }
        }
        return circles;
    }

    public static Set<Integer> nextLevel(Map<Integer, Set<Integer>> map, Set<Integer> subs,boolean[] visited) {
        Set<Integer> newSubs = new HashSet();
        for(Integer sub:subs){
            if(!map.containsKey(sub)){
                continue;
            }
            for(Integer node:map.get(sub)){
                // 已经访问过
                if(visited[node]){
                    continue;
                }
                newSubs.add(node);
                visited[node] = true;
            }
        }
        return newSubs;
    }


    public static void main(String[] args) {
        int[][] array = new  int[][]{{1, 1, 0}, {1, 1, 0}, {0, 0, 1}}; // 2
        // int[][] array = new int[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}; // 3;
        long ts0 = System.currentTimeMillis();
        System.out.println("meth 1 = " + findCircleNum(array));
        long ts1 = System.currentTimeMillis();
        System.out.println("time 1 = " + (ts1 - ts0) + "ms");
        System.out.println("meth 2 = " + bfs(array));
        long ts2 = System.currentTimeMillis();
        System.out.println("time 2 = " + (ts2 - ts1) + "ms");
        System.out.println("meth 3 = " + mergeFind(array));
        long ts3 = System.currentTimeMillis();
        System.out.println("time 3 = " + (ts3 - ts2) + "ms");
        System.out.println("meth 4 = " + map(array));
        long ts4 = System.currentTimeMillis();
        System.out.println("time 4 = " + (ts4 - ts3) + "ms");
    }
}
