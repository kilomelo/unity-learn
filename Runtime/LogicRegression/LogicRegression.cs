
using System;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

public class LogicRegression : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        // TestSigmoid();
        // TestPropagate();
        TestOptimize();
    }

    #region sigmoid
    /// <summary>
    /// Compute the sigmoid of z
    /// </summary>
    /// <param name="z">vector of any size.</param>
    /// <returns>sigmoid of z</returns>
    private Vector<double> Sigmoid(Vector<double> z)
    {
        return 1.0 / (1.0 + z.Negate().PointwiseExp());
    }
    private void TestSigmoid()
    {
        var testVector = Vector<double>.Build.Dense(new double[] {0.0, 2.0});
        Debug.Log(testVector);
        // output value: 0.5, 0.88079708
        Debug.Log(Sigmoid(testVector));
    }
    #endregion
    
    #region propagate
    /// <summary>
    /// Implement the cost function and its gradient for the propagation explained above
    /// </summary>
    /// <param name="w">weights, a numpy array of size (num_px * num_px * 3, 1)</param>
    /// <param name="b">bias, a scalar</param>
    /// <param name="x">data of size (num_px * num_px * 3, number of examples)</param>
    /// <param name="y">true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)</param>
    /// <param name="dw">gradient of the loss with respect to w, thus same shape as w</param>
    /// <param name="db">gradient of the loss with respect to b, thus same shape as b</param>
    /// <param name="cost">negative log-likelihood cost for logistic regression</param>
    /// <exception cref="ArgumentNullException"></exception>
    private void Propagate(Vector<double> w, double b, Matrix<double> x, Vector<double> y,
        out Vector<double> dw, out double db, out double cost)
    {
        if (null == w ||
            null == x ||
            null == y)
        {
            throw new ArgumentNullException();
        }
        var m = x.ColumnCount;
        
        // FORWARD PROPAGATION (FROM X TO COST)
        // A = sigmoid(np.dot(w.T, X) + b)
        // compute activation
        var a = Sigmoid(w * x + b);
        // compute cost
        // cost = -(1.0 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        cost = -(1.0 / m) * (a.PointwiseLog().PointwiseMultiply(y) + (1.0 - a).PointwiseLog().PointwiseMultiply(1.0 - y)).Sum();
        
        // BACKWARD PROPAGATION (TO FIND GRAD)
        // dw = (1.0 / m) * np.dot(X, (A - Y).T)
        dw = 1.0 / m * x * (a - y);
        // db = (1.0 / m) * np.sum(A - Y)
        db = 1.0 / m * (a - y).Sum();
    }

    private void TestPropagate()
    {
        var x = CreateMatrix.Dense(2, 3, new double[] {1.0, 3.0, 2.0, 4.0, -1.0, -3.2});
        var w = Vector<double>.Build.Dense(new double[] {1.0, 2.0});
        double b = 2.0;
        var y = Vector<double>.Build.Dense(new double[] {1.0, 0.0, 1.0});
        Propagate(w, b, x, y, out var dw, out var db, out var cost);
        Debug.Log($"dw: {dw}\n" +
                  $"db: {db}\n" +
                  $"cost: {cost}");
    }
    #endregion
    
    #region optimize

    private void Optimize(Vector<double> initialW, double initialB, Matrix<double> x, Vector<double> y,
        int numIterations, float learningRate,
        out Vector<double> w, out double b, out double cost, out Vector<double> dw, out double db)
    {
        if (numIterations <= 0) throw new ArgumentOutOfRangeException();
        
        w = initialW;
        b = initialB;
        cost = double.MaxValue;
        dw = w;
        db = b;
        for (var i = 0; i < numIterations; i++)
        {
            Propagate(w, b, x, y, out dw, out db, out cost);
            w -= learningRate * dw;
            b -= learningRate * db;
        }
    }

    private void TestOptimize()
    {
        var x = CreateMatrix.Dense(2, 3, new double[] {1.0, 3.0, 2.0, 4.0, -1.0, -3.2});
        var w = Vector<double>.Build.Dense(new double[] {1.0, 2.0});
        double b = 2.0;
        var y = Vector<double>.Build.Dense(new double[] {1.0, 0.0, 1.0});
        Optimize(w, b, x, y, 100, 0.009f,
            out var finalW, out var finalB, out var cost, out var dw, out var db);
        Debug.Log($"w: {finalW}\n" +
                  $"b: {finalB}\n" +
                  $"dw: {dw}\n" +
                  $"db: {db}");
    }
    #endregion
}
