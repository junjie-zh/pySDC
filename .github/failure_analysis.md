# Automated Test Failure Analysis

**Generated:** 2026-03-09T06:05:52.381490+00:00
**Workflow Run:** https://github.com/junjie-zh/pySDC/actions/runs/22840311461

## Summary

- Total Jobs: 30
- Failed Jobs: 2

## Failed Jobs

### 1. user_cpu_tests_linux (base, 3.13)

- **Job ID:** 66245025937
- **Started:** 2026-03-09T05:52:40Z
- **Completed:** 2026-03-09T05:55:34Z
- **Logs:** [View Job Logs](https://github.com/junjie-zh/pySDC/actions/runs/22840311461/job/66245025937)

#### Error Details

**Error 1:**
```
2026-03-09T05:53:54.0202351Z pySDC/tests/test_convergence_controllers/test_extrapolation_within_Q.py::test_extrapolation_within_Q[GAUSS-3] PASSED [ 15%]
2026-03-09T05:53:54.0459164Z pySDC/tests/test_convergence_controllers/test_extrapolation_within_Q.py::test_extrapolation_within_Q[GAUSS-4] PASSED [ 15%]
2026-03-09T05:53:54.1218987Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-2] FAILED [ 15%]
2026-03-09T05:53:54.1513667Z pySDC/test
```

**Error 2:**
```
2026-03-09T05:53:54.0459164Z pySDC/tests/test_convergence_controllers/test_extrapolation_within_Q.py::test_extrapolation_within_Q[GAUSS-4] PASSED [ 15%]
2026-03-09T05:53:54.1218987Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-2] FAILED [ 15%]
2026-03-09T05:53:54.1513667Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-3] FAILED [ 15%]
2026-03-09T05:53:54.1806339Z pySDC/te
```

**Error 3:**
```
2026-03-09T05:53:54.1218987Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-2] FAILED [ 15%]
2026-03-09T05:53:54.1513667Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-3] FAILED [ 15%]
2026-03-09T05:53:54.1806339Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-4] FAILED [ 15%]
2026-03-09T05:53:54.2098436Z pySDC/
```

**Error 4:**
```
2026-03-09T05:53:54.1513667Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-3] FAILED [ 15%]
2026-03-09T05:53:54.1806339Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-4] FAILED [ 15%]
2026-03-09T05:53:54.2098436Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-5] FAILED [ 15%]
2026-03-09T05:53:54.2390257Z pySDC/
```

**Error 5:**
```
2026-03-09T05:53:54.1806339Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-4] FAILED [ 15%]
2026-03-09T05:53:54.2098436Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-5] FAILED [ 15%]
2026-03-09T05:53:54.2390257Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-GAUSS-2] FAILED [ 15%]
2026-03-09T05:53:54.2687777Z pySDC/tests/
```

### 2. user_cpu_tests_linux (base, 3.11)

- **Job ID:** 66245025949
- **Started:** 2026-03-09T05:52:39Z
- **Completed:** 2026-03-09T05:55:34Z
- **Logs:** [View Job Logs](https://github.com/junjie-zh/pySDC/actions/runs/22840311461/job/66245025949)

#### Error Details

**Error 1:**
```
2026-03-09T05:53:53.3149195Z pySDC/tests/test_convergence_controllers/test_extrapolation_within_Q.py::test_extrapolation_within_Q[GAUSS-3] PASSED [ 15%]
2026-03-09T05:53:53.3404557Z pySDC/tests/test_convergence_controllers/test_extrapolation_within_Q.py::test_extrapolation_within_Q[GAUSS-4] PASSED [ 15%]
2026-03-09T05:53:53.4379714Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-2] FAILED [ 15%]
2026-03-09T05:53:53.4661051Z pySDC/test
```

**Error 2:**
```
2026-03-09T05:53:53.3404557Z pySDC/tests/test_convergence_controllers/test_extrapolation_within_Q.py::test_extrapolation_within_Q[GAUSS-4] PASSED [ 15%]
2026-03-09T05:53:53.4379714Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-2] FAILED [ 15%]
2026-03-09T05:53:53.4661051Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-3] FAILED [ 15%]
2026-03-09T05:53:53.4944632Z pySDC/te
```

**Error 3:**
```
2026-03-09T05:53:53.4379714Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-2] FAILED [ 15%]
2026-03-09T05:53:53.4661051Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-3] FAILED [ 15%]
2026-03-09T05:53:53.4944632Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-4] FAILED [ 15%]
2026-03-09T05:53:53.5233372Z pySDC/
```

**Error 4:**
```
2026-03-09T05:53:53.4661051Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-3] FAILED [ 15%]
2026-03-09T05:53:53.4944632Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-4] FAILED [ 15%]
2026-03-09T05:53:53.5233372Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-5] FAILED [ 15%]
2026-03-09T05:53:53.5517935Z pySDC/
```

**Error 5:**
```
2026-03-09T05:53:53.4944632Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-4] FAILED [ 15%]
2026-03-09T05:53:53.5233372Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-RADAU-RIGHT-5] FAILED [ 15%]
2026-03-09T05:53:53.5517935Z pySDC/tests/test_convergence_controllers/test_polynomial_error.py::test_interpolation_error[True-GAUSS-2] FAILED [ 15%]
2026-03-09T05:53:53.5804752Z pySDC/tests/
```

## Recommended Actions

1. Review the error messages above
2. Check if this is a known issue in recent commits
3. Review the full logs linked above for complete context
4. Consider if this is related to:
   - Dependency updates (check recent dependency changes)
   - Environment configuration issues
   - Test infrastructure problems
   - Flaky tests that need to be fixed
5. If needed, manually investigate and apply fixes to this PR

## How to Use This PR

This PR was automatically created to help investigate test failures. You can:

- Use this PR to track the investigation
- Add commits with fixes directly to this branch
- Close this PR if the issue is resolved elsewhere
- Convert this to an issue if it needs more discussion
