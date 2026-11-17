$ErrorActionPreference = 'Stop'

$remote = 'connect.nmb2.seetacloud.com'
$testCommand = 'python -m pytest -q'
$runId = [guid]::NewGuid().ToString('N')
$archive = Join-Path ([System.IO.Path]::GetTempPath()) "codex-test-$runId.tar.gz"
$remoteArchive = "/root/autodl-tmp/codex-test-$runId.tar.gz"

try {
    tar.exe -czf $archive --exclude=.git --exclude=.venv --exclude=.env --exclude='.env.*' --exclude=__pycache__ --exclude=.pytest_cache -C $PSScriptRoot .
    if ($LASTEXITCODE -ne 0) { throw '本地代码打包失败。' }

    scp -o BatchMode=yes $archive "${remote}:${remoteArchive}"
    if ($LASTEXITCODE -ne 0) { throw '同步失败，远端测试未运行。' }

    $remoteCommand = @'
archive='{0}'
work=$(mktemp -d /root/autodl-tmp/codex-test.XXXXXXXX) || exit 1
trap 'rm -f -- "$archive"; rm -rf -- "$work"' EXIT
tar -xzf "$archive" -C "$work" || exit 1
cd "$work" || exit 1
{1}
'@ -f $remoteArchive, $testCommand

    ssh -o BatchMode=yes $remote ($remoteCommand -replace "`r", '')
    $testExitCode = $LASTEXITCODE
}
finally {
    Remove-Item -LiteralPath $archive -ErrorAction SilentlyContinue
}

exit $testExitCode
