# In JavaScript V8 (Chrome / Edge / Node):
# The TimSort implementation compares elements using (b.prob - a.prob).
# If (b.prob - a.prob) is NaN, V8 treats the comparison as false or 0 depending on the check.
# Let's see: if a user passes an object with NaN, what happens in Chrome?
print("Checking what happens if z is undefined or NaN...")
