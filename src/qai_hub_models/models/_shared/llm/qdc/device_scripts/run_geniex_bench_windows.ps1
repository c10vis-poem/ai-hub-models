[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'

$ErrorActionPreference = "Continue"

$LOG = "C:\Temp\QDC_Logs"
$OUT = "$LOG\results"
$MM_CACHE = "C:\Temp\geniex-cache"
$TC = "C:\Temp\TestContent"

# QDC reuses device cells across jobs; wipe stale cell JSONs from a prior
# run so compute_metrics doesn't ingest another model/plugin's results.
Remove-Item -Recurse -Force $OUT -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $LOG, $OUT, $MM_CACHE | Out-Null
Start-Transcript -Path "$LOG\script.log" -Force | Out-Null

try {

# geniex-bench.exe writes informational lines to stderr even on success.
# Called via the bare `&` operator, every such line becomes a NativeCommandError
# ErrorRecord that QDC's parser flags as Unsuccessful — same trap that bit
# genie's run_windows.ps1. Start-Process redirects at the OS-process level so
# stderr bypasses PowerShell's error stream entirely. One retry mirrors
# Invoke-GenieRetry. Returns $true on success, $false on double-failure so the
# caller can defer the fatal exit until after the accuracy eval.
function Invoke-GenieXBenchRetry {
    param([Parameter(Mandatory = $true)][string[]]$BenchArgs)
    foreach ($attempt in 1, 2) {
        $stdoutFile = [System.IO.Path]::GetTempFileName()
        $stderrFile = [System.IO.Path]::GetTempFileName()
        try {
            $proc = Start-Process -FilePath "$BUNDLE\bin\geniex-bench.exe" `
                -ArgumentList $BenchArgs `
                -NoNewWindow -Wait -PassThru `
                -RedirectStandardOutput $stdoutFile -RedirectStandardError $stderrFile
            $exitCode = $proc.ExitCode
            $stdout = Get-Content $stdoutFile -Raw -Encoding UTF8
            $stderr = Get-Content $stderrFile -Raw -Encoding UTF8
        } finally {
            Remove-Item $stdoutFile -ErrorAction SilentlyContinue
            Remove-Item $stderrFile -ErrorAction SilentlyContinue
        }
        $captured = @($stdout) + @($stderr)
        $captured | Out-String | Write-Host
        if ($exitCode -eq 0) { return $true }
        Write-Host "Invoke-GenieXBenchRetry: geniex-bench.exe failed (exit $exitCode)"
    }
    return $false
}

$ZIP = "$TC\geniex-bench.zip"
$URL = "{WINDOWS_BENCH_URL}"
& curl.exe -fSL --retry 3 --retry-delay 5 -o $ZIP $URL
if ($LASTEXITCODE -ne 0) { throw "geniex-bench download failed: $LASTEXITCODE" }
Expand-Archive -Path $ZIP -DestinationPath $TC -Force
Remove-Item $ZIP

$BUNDLE = (Get-ChildItem -Path $TC -Directory -Filter 'geniex-bench-windows-arm64-*' | Select-Object -First 1).FullName
if (-not $BUNDLE) { throw "extracted bundle dir missing under $TC" }
if (-not (Test-Path "$BUNDLE\bin\geniex-bench.exe")) {
    throw "geniex-bench.exe not found at $BUNDLE\bin"
}

Set-Location $BUNDLE
$env:GENIEX_PLUGIN_PATH = "$BUNDLE\lib"
$env:PATH = "$BUNDLE\lib;$BUNDLE\lib\llama_cpp;$BUNDLE\lib\qairt;$BUNDLE\lib\qairt\htp-files;$env:PATH"

$rows = @'
{MODELS}
'@ -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ }

$IMG = "$TC/test.png" -replace '\\', '/'

$RUN_PERF = {RUN_PERF}
$RUN_EVAL = {RUN_EVAL}
$ctxList = @({CTX_LIST})
# Perf records failures but defers the fatal exit to the end so a perf failure
# never skips the accuracy eval below.
$failed_ctxs = ""

if ($RUN_PERF) {
    $tsvByCtx = @{}
    foreach ($ctx in $ctxList) {
        $tsvByCtx[$ctx] = "C:\Temp\matrix-$ctx.tsv"
        Remove-Item $tsvByCtx[$ctx] -ErrorAction SilentlyContinue
    }

    foreach ($row in $rows) {
        $name, $plugin, $devs, $model_id, $vlm, $image = $row -split '\|'
        Write-Output "=== plan $name id=$model_id ==="
        $imgpath = if ($image -eq "1") { $IMG } else { "" }
        foreach ($d in $devs -split ',') {
            foreach ($ctx in $ctxList) {
                "{0}-{1}-{2}-c{3}`t{1}`t{2}`t{4}`t`t`t{5}`t{6}" -f `
                    $name, $plugin, $d, $ctx, $model_id, $imgpath, $vlm `
                    | Add-Content $tsvByCtx[$ctx]
            }
        }
    }

    foreach ($ctx in $ctxList) {
        $tsv = $tsvByCtx[$ctx]
        Write-Output "=== matrix ctx=$ctx ==="
        if (Test-Path $tsv) { Get-Content $tsv }
        $ok = Invoke-GenieXBenchRetry -BenchArgs @(
            "--matrix-file", $tsv, "--output-json-dir", $OUT, "-r", "3",
            {BENCH_SIZE_FLAGS_ARGS}
            "--mm-data-dir", $MM_CACHE, "--chipset", "{CHIPSET}"
        )
        if (-not $ok) { $failed_ctxs += " $ctx" }
        Write-Output "$((Get-ChildItem $OUT).Count) cell json files so far"
    }
} else {
    Write-Output "=== perf sweep disabled (RUN_PERF=`$false); eval only ==="
}

# Accuracy eval: runs `geniex-bench --accuracy` once per staged prompt, marking
# each invocation's stdout with ===EVAL_IDX_NNN=== for the host to split.
$PROMPT_DIR = "$TC\prompts"
$EVAL_OUT = "$LOG\geniex_eval_outputs.txt"
if ($RUN_EVAL) {
    if (-not (Test-Path $PROMPT_DIR)) {
        throw "eval requested but $PROMPT_DIR missing"
    }
    Write-Output "=== accuracy eval ==="
    # Eval model/plugin/device come from the first matrix row (single-model job).
    # @($rows) forces array semantics: with a single MODELS line the pipeline
    # that builds $rows unwraps to a scalar string, so a bare $rows[0] would
    # index the first character instead of the first row.
    $e_name, $e_plugin, $e_devs, $e_model = (@($rows)[0] -split '\|')[0..3]
    $e_dev = "npu"
    $EVAL_ERR = "$LOG\geniex_eval_stderr.txt"
    Remove-Item $EVAL_OUT, $EVAL_ERR -ErrorAction SilentlyContinue
    $eval_ran = 0
    foreach ($pf in Get-ChildItem "$PROMPT_DIR\prompt_*.txt" | Sort-Object Name) {
        $idx = $pf.BaseName -replace 'prompt_', ''
        $tmpOut = $EVAL_OUT.Replace('.txt', '.tmp')
        $tmpErr = $EVAL_ERR.Replace('.txt', '.tmp')
        # Retry a random device flake once, but not a "Context length exceeded"
        # overflow -- that's a legit long generation, so keep the partial output
        # and stop. Each attempt writes to fresh temp files; the accepted
        # attempt's stdout is appended under the ===EVAL_IDX=== marker exactly
        # once so retries do not concatenate two generations at the grader.
        $exitCode = 1
        $stderrText = ""
        for ($attempt = 1; $attempt -le 2; $attempt++) {
            # Wipe any leftover output from the previous attempt so this loop
            # iteration only ever contributes one attempt's stdout/stderr.
            Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue
            try {
                # Start-Process redirects at the OS-process level so geniex-bench's
                # stderr info lines don't become PowerShell NativeCommandErrors (QDC
                # flags those Unsuccessful).
                $proc = Start-Process -FilePath "$BUNDLE\bin\geniex-bench.exe" -ArgumentList @(
                    "--plugin", $e_plugin, "--device", $e_dev, "-m", $e_model,
                    "--accuracy", "--prompt-file", $pf.FullName,
                    "-c", "{EVAL_CTX}", "-n", "{EVAL_N_GEN}",
                    "--mm-data-dir", $MM_CACHE, "--chipset", "{CHIPSET}"
                ) -NoNewWindow -PassThru -RedirectStandardOutput $tmpOut `
                    -RedirectStandardError $tmpErr
                # timeout kills a crashed/hung DSP run so the loop advances.
                if (-not $proc.WaitForExit({EVAL_TIMEOUT_S} * 1000)) {
                    Write-Host "eval idx $idx timed out after {EVAL_TIMEOUT_S}s; killing"
                    try { $proc.Kill($true) } catch { $proc.Kill() }
                    $proc.WaitForExit()
                }
                $exitCode = $proc.ExitCode
                $stderrText = Get-Content $tmpErr -Raw -Encoding UTF8
            } catch {
                Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue
                throw
            }
            if ($exitCode -eq 0 -or ($stderrText -match 'Context length exceeded')) { break }
            if ($attempt -lt 2) {
                Write-Host "eval idx $idx failed (exit $exitCode); retrying once"
            }
        }
        # Emit the marker + accepted attempt's output only once, then clean up.
        try {
            Add-Content -Path $EVAL_OUT -Value "===EVAL_IDX_$idx==="
            Add-Content -Path $EVAL_ERR -Value "===EVAL_IDX_$idx==="
            if (Test-Path $tmpOut) { Get-Content $tmpOut -Encoding UTF8 | Add-Content $EVAL_OUT }
            if (Test-Path $tmpErr) { Get-Content $tmpErr -Encoding UTF8 | Add-Content $EVAL_ERR }
        } finally {
            Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue
        }
        # A context-length overflow counts as run (keeps its partial output).
        if ($exitCode -eq 0 -or ($stderrText -match 'Context length exceeded')) {
            $eval_ran++
        } else {
            Write-Host "eval idx $idx failed (exit $exitCode)"
        }
        Start-Sleep -Seconds {EVAL_SLEEP_S}
    }
    $n = (Select-String -Path $EVAL_OUT -Pattern '===EVAL_IDX_').Count
    Write-Output "eval done ($n prompts, $eval_ran ran)"
    if ($eval_ran -eq 0) {
        throw "accuracy eval produced no output"
    }
}

Write-Output "=== done ==="

# Deferred perf failure: raise inside the try so the finally cleanup still runs.
if ($failed_ctxs) { throw "geniex-bench failed for context lengths:$failed_ctxs" }

}
finally {
    # Drop per-job state on exit (dedicated-pool devices are reused across jobs;
    # leftover extracted bundles / caches / matrix TSVs would leak into the next
    # tenant's run).
    Get-ChildItem -Path $TC -Directory -Filter 'geniex-bench-windows-arm64-*' -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Force "$TC\geniex-bench.zip" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $MM_CACHE -ErrorAction SilentlyContinue
    Get-ChildItem -Path "C:\Temp" -Filter 'matrix-*.tsv' -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Stop-Transcript | Out-Null
}
exit 0
