package lesson.tree;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 * 二叉树的最小深度
 */
public class Lesson6 {

    /**
     * 解法1：深度优先
     * 遍历整颗数，找到每一个叶子节点，从叶子节点往上开始计算，左右子节点都为空则记录深度为1
     * 左右子节点只有一边，深度记录为子节点深度+1
     * 左右两边都有子节点，则记录左右子节点的深度较小值+1
     */
    public static int minDepthDFS(TreeNode root) {
        if (root == null) {
            return 0;
        }
        if (root.left == null && root.right == null) {
            return 1;
        }
        int min_depth = Integer.MAX_VALUE;
        if (root.left != null) {
            min_depth = Math.min(minDepthDFS(root.left), min_depth);
        }
        if (root.right != null) {
            min_depth = Math.min(minDepthDFS(root.right), min_depth);
        }
        return min_depth + 1;
    }



    /**
     * 解法2：解法二：广度优先
     * 从上往下，找到一个节点时，标记这个节点的深度。查看该节点是否为叶子节点，如果是直接返回深度
     * 如果不是叶子节点，将其子节点标记深度(在父节点深度的基础上加1)，再判断该节点是否为叶子节点
     */
    static class QueueNode {
        TreeNode node;
        int depth;
        public QueueNode(TreeNode node, int depth) {
            this.node = node;
            this.depth = depth;
        }
    }
    public static int minDepthBFS(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<QueueNode> queue = new LinkedList<QueueNode>();
        queue.offer(new QueueNode(root, 1));
        while (!queue.isEmpty()) {
            QueueNode nodeDepth = queue.poll();
            TreeNode node = nodeDepth.node;
            int depth = nodeDepth.depth;
            if (node.left == null && node.right == null) {
                return depth;
            }
            if (node.left != null) {
                queue.offer(new QueueNode(node.left, depth + 1));
            }
            if (node.right != null) {
                queue.offer(new QueueNode(node.right, depth + 1));
            }
        }
        return 0;
    }

    /**
     * 解法3：广度优先
     * 从根节点开始逐层查看是否出现叶子节点
     */
    public static int minDepthBFS2(TreeNode root) {
        int deep = 0;
        // 根节点不存在
        if (root == null) {
            return deep;
        }

        List<TreeNode> subs = new ArrayList();
        subs.add(root);
        while (subs.size() > 0){
            subs = nextLevel(subs);
            deep ++;
        }
        return deep;
    }

    /**
     * 没有出现叶子节点时返回全部下一层，否则返回空
     */
    public static List<TreeNode> nextLevel(List<TreeNode> subs) {
        List<TreeNode> newSubs = new ArrayList();
        for(TreeNode root:subs){
            // 叶子节点
            if (root.left == null && root.right == null) {
                return new ArrayList();
            }
            // 添加下一层
            if (root.left != null) {
                newSubs.add(root.left);
            }
            if (root.right != null) {
                newSubs.add(root.right);
            }
        }
        return newSubs;
    }


    public static void main(String[] args) {
        TreeNode root = new TreeNode().creteTree();
        long ts0 = System.currentTimeMillis();
        System.out.println("meth 1 = " + minDepthDFS(root));
        long ts1 = System.currentTimeMillis();
        System.out.println("time 1 = " + (ts1 - ts0) + "ms");
        System.out.println("meth 2 = " + minDepthBFS(root));
        long ts2 = System.currentTimeMillis();
        System.out.println("time 2 = " + (ts2 - ts1) + "ms");
        System.out.println("meth 3 = " + minDepthBFS2(root));
        long ts3 = System.currentTimeMillis();
        System.out.println("time 3 = " + (ts3 - ts2) + "ms");
    }
}
