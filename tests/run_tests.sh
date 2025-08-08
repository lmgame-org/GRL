#!/bin/bash

# =============================================================================
# LMGameRL Test Runner
# Runs all tests in the tests folder and logs output to test_logs directory
# =============================================================================

set -uo pipefail  # Don't exit on error, let us handle them gracefully

# Maximum number of parallel tests
MAX_PARALLEL_TESTS=${MAX_PARALLEL_TESTS:-3}

# Continue on test failures
CONTINUE_ON_FAIL=${CONTINUE_ON_FAIL:-1}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "\n${CYAN}===============================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}===============================================${NC}\n"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_test_result() {
    local test_name="$1"
    local status="$2"
    local duration="$3"
    
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}✓${NC} $test_name ${CYAN}(${duration}s)${NC}"
    elif [ "$status" = "FAIL" ]; then
        echo -e "${RED}✗${NC} $test_name ${CYAN}(${duration}s)${NC}"
    elif [ "$status" = "TIMEOUT" ]; then
        echo -e "${RED}⏰${NC} $test_name ${CYAN}(${duration}s)${NC}"
    elif [ "$status" = "SKIP" ]; then
        echo -e "${YELLOW}⊘${NC} $test_name ${CYAN}(skipped)${NC}"
    fi
}

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TESTS_DIR="$SCRIPT_DIR"

# Create main test logs directory
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
MAIN_LOG_DIR="$TESTS_DIR/test_logs"
SESSION_LOG_DIR="$MAIN_LOG_DIR/test_session_$TIMESTAMP"
mkdir -p "$SESSION_LOG_DIR"

# Main log file for the entire test session
MAIN_LOG_FILE="$SESSION_LOG_DIR/test_session.log"

# Initialize main log
exec > >(tee -a "$MAIN_LOG_FILE")
exec 2>&1

# Test session variables
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0
declare -a FAILED_TEST_NAMES=()
declare -a RUNNING_PIDS=()
declare -a TEST_JOBS=()

print_header "lmgamerl Test Suite Runner"
echo "Started at: $(date)"
echo "Project root: $PROJECT_ROOT"
echo "Tests directory: $TESTS_DIR"
echo "Session logs: $SESSION_LOG_DIR"
echo "Main log: $MAIN_LOG_FILE"

# Function to run a single test file (with error isolation)
run_test() {
    local test_file="$1"
    local test_name="$2"
    
    # Create individual test log directory
    local test_log_dir="$SESSION_LOG_DIR/${test_name}"
    mkdir -p "$test_log_dir"
    
    local test_log_file="$test_log_dir/test.log"
    local test_error_file="$test_log_dir/error.log"
    local test_status_file="$test_log_dir/status.txt"
    
    # Run test in isolated environment to prevent crashes
    (
        exec > "$test_log_file" 2> "$test_error_file"
        
        local start_time=$(date +%s)
        echo "Starting test: $test_name at $(date)"
        echo "Test file: $test_file"
        echo "Working directory: $PROJECT_ROOT"
        echo "Python path: $PROJECT_ROOT"
        echo "=========================="
        
        # Change to project root for consistent imports
        cd "$PROJECT_ROOT" || exit 1
        
        # Set up isolated environment
        export PYTHONPATH="$PROJECT_ROOT"
        
        # Run the test with timeout and error handling
        local exit_code=0
        if timeout 600 python "$test_file"; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            echo "PASS:$duration" > "$test_status_file"
        else
            exit_code=$?
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            
            if [ $exit_code -eq 124 ]; then
                echo "TIMEOUT:$duration" > "$test_status_file"
            else
                echo "FAIL:$duration" > "$test_status_file"
            fi
        fi
    )
    
    # Wait a moment for file to be written
    sleep 0.5
    
    # Read test results
    local status="FAIL"
    local duration="0"
    
    if [ -f "$test_status_file" ]; then
        local status_line=$(cat "$test_status_file")
        status="${status_line%:*}"
        duration="${status_line#*:}"
    fi
    
    # Update counters and display results
    case "$status" in
        "PASS")
            print_test_result "$test_name" "PASS" "$duration"
            ((PASSED_TESTS++))
            
            # Copy any additional logs created by the test
            local test_logs_dir="$(dirname "$test_file")/test_logs"
            if [ -d "$test_logs_dir" ]; then
                cp -r "$test_logs_dir"/* "$test_log_dir/" 2>/dev/null || true
            fi
            ;;
        "TIMEOUT")
            print_test_result "$test_name" "TIMEOUT" "$duration"
            ((FAILED_TESTS++))
            FAILED_TEST_NAMES+=("$test_name (TIMEOUT)")
            ;;
        "FAIL"|*)
            print_test_result "$test_name" "FAIL" "$duration"
            ((FAILED_TESTS++))
            FAILED_TEST_NAMES+=("$test_name")
            
            # Show brief error info (non-blocking)
            if [ -s "$test_error_file" ]; then
                echo -e "${RED}Brief error:${NC}"
                head -n 3 "$test_error_file" | sed 's/^/  /' | head -c 200
                echo "..."
            fi
            ;;
    esac
    
    ((TOTAL_TESTS++))
}

# Function to run test in background
run_test_async() {
    local test_file="$1"
    local test_name="$2"
    
    print_step "Starting test: $test_name (async)"
    
    # Run test in background
    run_test "$test_file" "$test_name" &
    local pid=$!
    
    RUNNING_PIDS+=($pid)
    TEST_JOBS+=("$test_name:$pid")
    
    return 0
}

# Function to wait for running tests and manage parallel execution
wait_for_tests() {
    local max_wait=${1:-0}  # 0 means wait for all
    
    while [ ${#RUNNING_PIDS[@]} -gt $max_wait ]; do
        local new_pids=()
        local new_jobs=()
        
        for i in "${!RUNNING_PIDS[@]}"; do
            local pid=${RUNNING_PIDS[$i]}
            local job=${TEST_JOBS[$i]}
            
            if ! kill -0 $pid 2>/dev/null; then
                # Process finished
                wait $pid 2>/dev/null || true
            else
                # Process still running
                new_pids+=($pid)
                new_jobs+=("$job")
            fi
        done
        
        RUNNING_PIDS=("${new_pids[@]}")
        TEST_JOBS=("${new_jobs[@]}")
        
        if [ ${#RUNNING_PIDS[@]} -gt $max_wait ]; then
            sleep 1
        fi
    done
}

# Function to check if a test should be skipped
should_skip_test() {
    local test_file="$1"
    
    # Skip if file doesn't exist
    if [ ! -f "$test_file" ]; then
        return 0  # Skip
    fi
    
    # Skip __init__.py files
    if [[ "$(basename "$test_file")" == "__init__.py" ]]; then
        return 0  # Skip
    fi
    
    # Skip utility files
    if [[ "$(basename "$test_file")" == "rollout_test_utils.py" ]]; then
        return 0  # Skip
    fi
    
    # Skip if not a Python test file
    if [[ ! "$test_file" =~ \.py$ ]]; then
        return 0  # Skip
    fi
    
    # Skip verl tests (they're submodule tests)
    if [[ "$test_file" =~ /verl/ ]]; then
        return 0  # Skip
    fi
    
    # Skip external tests
    if [[ "$test_file" =~ /external/ ]]; then
        return 0  # Skip
    fi
    
    return 1  # Don't skip
}

# Function to discover and run all tests (with parallel execution)
discover_and_run_tests() {
    print_header "Test Discovery"
    
    # Find all Python test files, excluding verl and external directories
    local test_files=()
    while IFS= read -r -d '' file; do
        if ! should_skip_test "$file"; then
            test_files+=("$file")
        fi
    done < <(find "$TESTS_DIR" -name "*.py" -type f \( ! -path "*/verl/*" ! -path "*/external/*" \) -print0)
    
    # Sort test files for consistent ordering
    IFS=$'\n' test_files=($(sort <<<"${test_files[*]}"))
    unset IFS
    
    echo "Discovered ${#test_files[@]} test files:"
    for test_file in "${test_files[@]}"; do
        local rel_path="${test_file#$TESTS_DIR/}"
        echo "  - $rel_path"
    done
    echo ""
    
    if [ ${#test_files[@]} -eq 0 ]; then
        print_warning "No test files found!"
        return 1
    fi
    
    # Run each test with parallel execution
    print_header "Running Tests (max $MAX_PARALLEL_TESTS parallel)"
    
    for test_file in "${test_files[@]}"; do
        local rel_path="${test_file#$TESTS_DIR/}"
        local test_name="${rel_path//\//_}"  # Replace / with _
        test_name="${test_name%.py}"         # Remove .py extension
        
        # Wait if we have too many running tests
        wait_for_tests $((MAX_PARALLEL_TESTS - 1))
        
        # Start test asynchronously
        run_test_async "$test_file" "$test_name"
        
        # Small delay to prevent overwhelming
        sleep 0.5
    done
    
    # Wait for all remaining tests to complete
    print_step "Waiting for remaining tests to complete..."
    wait_for_tests 0
}

# Function to run specific test categories (parallel by test suite)
run_agent_tests() {
    print_header "Agent Tests (Parallel by Agent Type)"
    
    local agent_dirs=(
        "birdAgent_tests"
        "blocksworldAgent_tests"
        "gsm8kAgent_tests"
        "sokobanAgent_tests"
        "tetrisAgent_tests"
        "webshopAgent_tests"
    )
    
    # Function to run all tests for one agent type
    run_agent_suite() {
        local agent_dir="$1"
        local agent_path="$TESTS_DIR/$agent_dir"
        
        if [ ! -d "$agent_path" ]; then
            print_warning "Agent test directory not found: $agent_dir"
            return 1
        fi
        
        echo -e "\n${YELLOW}Running $agent_dir tests...${NC}"
        
        local tests_run=0
        
        # Run agent test
        if [ -f "$agent_path/agent_test.py" ]; then
            run_test "$agent_path/agent_test.py" "${agent_dir}_agent"
            ((tests_run++))
        else
            print_warning "No agent_test.py found in $agent_dir"
        fi
        
        # Run environment test
        if [ -f "$agent_path/env_test.py" ]; then
            run_test "$agent_path/env_test.py" "${agent_dir}_env"
            ((tests_run++))
        else
            print_warning "No env_test.py found in $agent_dir"
        fi
        
        return $tests_run
    }
    
    # Run each agent test suite in parallel
    for agent_dir in "${agent_dirs[@]}"; do
        # Wait if we have too many running test suites
        wait_for_tests $((MAX_PARALLEL_TESTS / 2))  # Use fewer parallel suites since each suite has multiple tests
        
        # Run the entire agent test suite in background
        (run_agent_suite "$agent_dir") &
        local pid=$!
        
        RUNNING_PIDS+=($pid)
        TEST_JOBS+=("${agent_dir}_suite:$pid")
        
        sleep 1  # Small delay between starting suites
    done
    
    # Wait for all agent test suites to complete
    print_step "Waiting for all agent test suites to complete..."
    wait_for_tests 0
}

# Function to run rollout tests (parallel)
run_rollout_tests() {
    print_header "Rollout Tests (Parallel)"
    
    local rollout_dir="$TESTS_DIR/rollout_tests"
    if [ -d "$rollout_dir" ]; then
        for test_file in "$rollout_dir"/*.py; do
            if [ -f "$test_file" ] && [[ "$(basename "$test_file")" != "__init__.py" ]] && [[ "$(basename "$test_file")" != "rollout_test_utils.py" ]]; then
                # Wait if we have too many running tests
                wait_for_tests $((MAX_PARALLEL_TESTS - 1))
                
                local test_name="rollout_$(basename "$test_file" .py)"
                run_test_async "$test_file" "$test_name"
                
                sleep 0.5
            fi
        done
        
        # Wait for all rollout tests to complete
        print_step "Waiting for rollout tests to complete..."
        wait_for_tests 0
    else
        print_warning "Rollout test directory not found"
    fi
}

# Function to generate test report
generate_report() {
    local report_file="$SESSION_LOG_DIR/test_report.txt"
    local summary_file="$SESSION_LOG_DIR/test_summary.txt"
    
    print_header "Test Results Summary"
    
    # Generate detailed report
    {
        echo "lmgamerl Test Suite Report"
        echo "=========================="
        echo "Generated: $(date)"
        echo "Session: $TIMESTAMP"
        echo ""
        echo "Test Statistics:"
        echo "  Total tests: $TOTAL_TESTS"
        echo "  Passed: $PASSED_TESTS"
        echo "  Failed: $FAILED_TESTS"
        echo "  Skipped: $SKIPPED_TESTS"
        echo ""
        
        if [ ${#FAILED_TEST_NAMES[@]} -gt 0 ]; then
            echo "Failed Tests:"
            for failed_test in "${FAILED_TEST_NAMES[@]}"; do
                echo "  - $failed_test"
            done
            echo ""
        fi
        
        echo "Log Files:"
        echo "  Main log: $MAIN_LOG_FILE"
        echo "  Session directory: $SESSION_LOG_DIR"
        echo ""
        
    } > "$report_file"
    
    # Generate summary
    {
        echo "TOTAL: $TOTAL_TESTS | PASSED: $PASSED_TESTS | FAILED: $FAILED_TESTS | SKIPPED: $SKIPPED_TESTS"
    } > "$summary_file"
    
    # Display summary
    cat "$report_file"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        print_success "All tests passed! 🎉"
        return 0
    else
        print_error "$FAILED_TESTS test(s) failed!"
        echo ""
        echo "Check individual test logs in: $SESSION_LOG_DIR"
        return 1
    fi
}

# Main execution
main() {
    # Check prerequisites
    if [ ! -d "$PROJECT_ROOT/lmgamerl" ]; then
        print_error "lmgamerl package not found at $PROJECT_ROOT/lmgamerl"
        print_error "Please ensure you're running this from the correct directory"
        exit 1
    fi
    
    # Setup Python path
    export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
    
    # Check if we're in the right environment
    if ! python -c "import sys; print(sys.version)" >/dev/null 2>&1; then
        print_error "Python not available"
        exit 1
    fi
    
    print_step "Python version: $(python --version 2>&1)"
    print_step "Python path: $PYTHONPATH"
    print_step "Working directory: $(pwd)"
    
    # Try to import lmgamerl to verify setup
    if python -c "import lmgamerl" 2>/dev/null; then
        print_success "lmgamerl package can be imported"
    else
        print_warning "lmgamerl package cannot be imported - some tests may fail"
        print_warning "Consider running: pip install -e ."
    fi
    echo ""
    
    # Parse command line arguments
    case "${1:-all}" in
        "agents"|"agent")
            run_agent_tests
            ;;
        "rollout"|"rollouts")
            run_rollout_tests
            ;;
        "all"|"")
            discover_and_run_tests
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [agents|rollout|all|help]"
            echo ""
            echo "Options:"
            echo "  agents   - Run only agent tests (parallel by agent type)"
            echo "  rollout  - Run only rollout tests (parallel)"
            echo "  all      - Run all tests (parallel, default)"
            echo "  help     - Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  MAX_PARALLEL_TESTS=N    - Set max parallel tests (default: 3)"
            echo "  CONTINUE_ON_FAIL=1      - Continue running tests after failures (default: 1)"
            echo "  VERBOSE=1               - More detailed output"
            echo ""
            echo "Examples:"
            echo "  $0                      # Run all tests with default settings"
            echo "  MAX_PARALLEL_TESTS=5 $0 agents  # Run agent tests with 5 parallel jobs"
            echo "  $0 rollout              # Run only rollout tests"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
    
    # Generate final report
    generate_report
}

# Cleanup function (with process cleanup)
cleanup() {
    # Kill any remaining background processes
    if [ ${#RUNNING_PIDS[@]} -gt 0 ]; then
        print_step "Cleaning up remaining test processes..."
        for pid in "${RUNNING_PIDS[@]}"; do
            if kill -0 $pid 2>/dev/null; then
                kill $pid 2>/dev/null || true
            fi
        done
        wait 2>/dev/null || true
    fi
    
    cd "$PROJECT_ROOT" 2>/dev/null || true
    print_step "Test session completed at: $(date)"
    echo "Full logs available at: $SESSION_LOG_DIR"
}

# Set up cleanup trap
trap cleanup EXIT

# Run main function
main "$@"
